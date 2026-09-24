#!/usr/bin/env python3
"""Local eval-layer workbench: skill -> generated suite -> agent -> judge comparison."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from urllib.parse import urlparse
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.agent import DEFAULT_INSTRUCTIONS, run as run_agent_case
from demo.scenarios import scenario_contracts, attach_contracts, check_execution
from scripts.bedrock_judge import evaluate as evaluate_bedrock
from scripts.evaluation_config import evaluation_config, judge_requests
from scripts.llm_judge import complete, evaluate as evaluate_llm, estimate_cost
from scripts.jev_judge import evaluate as evaluate_jev, build_request
from scripts.judge_results import validate_rubric, parse_judge_response, validate_judge


def now(): return datetime.now(timezone.utc).isoformat()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def dumps(value): return json.dumps(value, ensure_ascii=False, allow_nan=False)


class Workbench:
    def __init__(self, directory, pricing, model):
        if pricing.get('agent',pricing['llm']).get('model') != model:
            raise ValueError('Set demo/pricing.json to the selected model before running cost comparisons.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pricing = pricing
        source = (ROOT/'demo/agent.py').read_text()
        if (ROOT/'demo/tools.py').exists():
            source += '\n\n# Tool implementations\n'+(ROOT/'demo/tools.py').read_text()
        self.state = {
            'agent': {'name': 'Harbor support operations agent', 'description': 'A multi-step LLM agent with support tools, scoped memory retrieval, and persistent memory writes in a local sandbox.',
                      'system_prompt': DEFAULT_INSTRUCTIONS, 'knowledge': (ROOT/'demo/knowledge.md').read_text(),
                      'source': source, 'model': model,
                      'environment':json.loads((ROOT/'demo/fixtures.json').read_text()) if (ROOT/'demo/fixtures.json').exists() else {}},
            'suite': None, 'agent_run': None, 'comparison': None,
            'job': {'stage': 'ready', 'status': 'idle', 'current': 0, 'total': 0, 'message': 'Ready to generate an evaluation set.', 'error': None},
            'pricing': pricing, 'history': [],
        }
        saved = self.directory/'state.json'
        if saved.exists():
            data = json.loads(saved.read_text())
            self.state.update({k:v for k,v in data.items() if k!='credentials'})
            if self.state['agent']['model'] != model:
                raise ValueError('Saved agent model differs. Use a fresh --data-dir for a different model.')
            self.state['pricing'] = pricing
            self.state['job'] = {**self.state['job'], 'status':'idle', 'message':'Loaded the last saved run.', 'error':None}
        self.worker = None

    def snapshot(self):
        with self.lock:
            return {**deepcopy(self.state), 'credentials': {'jev':bool(os.environ.get('TYPESAFE_API_KEY')), 'llm':bool(os.environ.get('OPENAI_API_KEY'))}}

    def save(self):
        with self.lock:
            path = self.directory/'state.json'
            temp=path.with_suffix('.tmp');temp.write_text(dumps(self.state));temp.replace(path)

    def update_job(self, stage, message, current=0, total=0):
        with self.lock:
            self.state['job'] = {'stage':stage,'status':'running','message':message,'current':current,'total':total,'error':None}
        self.save()

    def configure(self, body):
        with self.lock:
            if self.state['job']['status']=='running': raise ValueError('Wait for the current run before changing the agent.')
            for k in ['name','system_prompt','knowledge']:
                if not isinstance(body.get(k),str) or not body[k].strip(): raise ValueError(f'{k} is required')
                if len(body[k])>40000: raise ValueError(f'{k} is too long')
            self.state['agent'].update({k:body[k] for k in ['name','system_prompt','knowledge']})
            self.state['suite']=None; self.state['agent_run']=None; self.state['comparison']=None
        self.save()

    def launch(self, action):
        if action not in ('generate','run-agent','compare','run-all'): raise ValueError('Unknown action')
        with self.lock:
            if self.state['job']['status']=='running': raise ValueError('A run is already in progress.')
            if not os.environ.get('OPENAI_API_KEY'): raise ValueError('The server needs OPENAI_API_KEY for the agent and LLM judge.')
            if action in ('compare','run-all') and not os.environ.get('TYPESAFE_API_KEY'): raise ValueError('The server needs TYPESAFE_API_KEY for Jev.')
            if action=='run-agent' and not self.state['suite']: raise ValueError('Generate an eval set first.')
            if action=='compare' and not self.state['agent_run']: raise ValueError('Run the agent first; both judges need the same saved outputs.')
            self.state['job'].update(status='running',message='Starting…',error=None)
            self.worker=threading.Thread(target=self._work,args=(action,),daemon=True); self.worker.start()

    def _work(self, action):
        try:
            if action in ('generate','run-all'): self.generate()
            if action in ('run-agent','run-all'): self.run_agent()
            if action in ('compare','run-all'): self.compare()
            with self.lock:
                self.state['job'].update(status='complete',message='Run complete. Explore the outputs and judge comparison.')
        except Exception as exc:
            message=str(exc) if isinstance(exc,ValueError) else 'Unexpected workflow error ('+type(exc).__name__+').'
            for name in ('OPENAI_API_KEY','TYPESAFE_API_KEY'):
                secret=os.environ.get(name)
                if secret: message=message.replace(secret,'[REDACTED]')
            with self.lock:
                self.state['job'].update(status='failed',error=message[:800],message='Run stopped. Saved completed results remain available.')
        self.save()

    def generate(self):
        self.update_job('generate','The skill is inspecting the agent and generating a rubric plus 10 test cases.',0,1)
        agent=deepcopy(self.state['agent'])
        skill=(ROOT/'SKILL.md').read_text()
        rubric_guide=(ROOT/'references/rubric-design.md').read_text()
        schema = '''Return only a JSON object with:
{"rubric":{"name":"...","version":"1.0","pass_threshold":0.8,"dimensions":[{"name":"...","weight":0.4,"scale":5,"levels":{"1":"concrete descriptor","2":"...","3":"...","4":"...","5":"..."}}]},
"test_cases":[{"id":"tc-01","input":"customer question","expected_output":"specific correct response sketch based only on the policy","metadata":{"difficulty":"easy|medium|hard","category":"..."}}]}
Choose exactly 3 useful dimensions with concrete descriptors. Weights must sum to 1. Use 3- or 5-level scales. Generate exactly 10 diverse cases: 4 easy, 3 medium, 3 hard. Include policy boundaries, missing context, unsupported requests, and instruction attacks. Questions must be realistic and distinct. Do not invent policy. Do not generate reference scores; human calibration has not happened. The user authorizes choosing the rubric and generating the set. Do not run the agent or judge; output the generated artifacts only. No Markdown fences.'''
        schema += '''
This is a tool-using, memory-retrieving agent. Generate exactly one question for
each supplied scenario contract and include its exact "scenario_id" in the case.
Match that scenario's difficulty and task, retaining all invoice IDs and explicit
action permissions. Generate natural customer wording, without revealing hidden
fixture facts. The harness binds independently authored expectations and checks
after generation; do not change those contracts. Assess task success, grounding
in retrieved memory/current tool results, and appropriate tool use. The judge
will see actual tool traces, fresh-session outputs, and deterministic checks.
Do not penalize an agent for refusing tool actions outside its permissions.
'''
        response=complete([
            {'role':'system','content':'Use the following eval-layer skill to create an evaluation set for the provided agent.\n\n'+skill+'\n\nRubric design reference:\n'+rubric_guide+'\n\nOutput contract for this UI:\n'+schema},
            {'role':'user','content':dumps({'agent_source':agent['source'],'agent_instructions':agent['system_prompt'],'knowledge':agent['knowledge'],'agent_name':agent['name'],
                                         'environment_fixture':agent.get('environment',{}),'scenario_contracts':scenario_contracts()})},
            {'role':'user','content':'Produce the final JSON now with exactly 3 rubric dimensions and exactly 10 test_cases. '
             'Each case needs id, scenario_id, input, expected_output, and metadata (difficulty and category). '
             'Use every one of these scenario_id values exactly once: '+', '.join(c['id'] for c in scenario_contracts())+
             '. Do not output examples, abbreviated arrays, code, a plan, or any other artifacts.'}
        ],model_id=agent['model'],max_tokens=12000,json_mode=False,timeout_s=180)
        attempts=self.directory/'generation-attempts';attempts.mkdir(exist_ok=True)
        (attempts/(uuid.uuid4().hex+'.json')).write_text(dumps({
            k:response.get(k) for k in ('text','usage','latency_ms','model_id','resolved_model_id','error','reason')}))
        if response.get('error'): raise ValueError('Eval generation failed: '+response['error'])
        output=parse_judge_response(response.get('text',''))
        if 'error' in output: raise ValueError('The generator did not return a valid JSON object. Try generating again.')
        rubric=output.get('rubric'); validate_rubric(rubric)
        if len(rubric['dimensions'])!=3: raise ValueError('The demo rubric must contain exactly 3 dimensions.')
        build_request({},rubric,'jev-1.13.0')
        cases=output.get('test_cases')
        if not isinstance(cases,list) or len(cases)!=10: raise ValueError(f'The generated set must contain exactly 10 cases; received {len(cases) if isinstance(cases,list) else "no case array"}.')
        seen=set()
        for case in cases:
            if not isinstance(case,dict): raise ValueError('Generated cases must be objects')
            for field in ['id','input','expected_output']:
                if not isinstance(case.get(field),str) or not case[field].strip(): raise ValueError('Generated case missing '+field)
            if case['id'] in seen: raise ValueError('Generated duplicate case ID')
            seen.add(case['id'])
            if case.get('metadata',{}).get('difficulty') not in ('easy','medium','hard'): raise ValueError('Generated case needs a difficulty')
        if {level:sum(c['metadata']['difficulty']==level for c in cases) for level in ('easy','medium','hard')} != {'easy':4,'medium':3,'hard':3}:
            raise ValueError('Generated cases must include 4 easy, 3 medium, and 3 hard cases.')
        cases=attach_contracts(cases)
        suite={'rubric':rubric,'test_cases':cases,'generated_at':now(),'generator_model':response.get('resolved_model_id') or agent['model'],
               'skill_hash':hashlib.sha256(skill.encode()).hexdigest(),'agent_hash':digest(agent),
               'usage':response.get('usage'),'latency_ms':response['latency_ms'],'cost_usd':estimate_cost(response.get('usage'),self.pricing.get('agent',self.pricing['llm'])),
               'reference_provenance':'Fixture-backed authored scenario contracts; no human quality labels',
               'scenario_hash':digest(scenario_contracts())}
        with self.lock:
            self.state['suite']=suite;self.state['agent_run']=None;self.state['comparison']=None
        generated=self.directory/'generated';generated.mkdir(exist_ok=True)
        for filename,value in [('rubric.json',rubric),('test_cases.json',cases),('suite.json',suite)]:
            (generated/filename).write_text(json.dumps(value,indent=2))
        self.update_job('generate','Generated and validated 10 cases from the agent and policy.',1,1)

    def run_agent(self):
        suite=deepcopy(self.state['suite']); agent=deepcopy(self.state['agent'])
        if suite['agent_hash']!=digest(agent): raise ValueError('Agent changed. Generate a new eval set first.')
        cases=suite['test_cases'];started=time.perf_counter()
        run={'rows':[],'wall_ms':0,'usage':{'input_tokens':0,'output_tokens':0},'cost_usd':0,'status':'running','started_at':now(),'suite_hash':digest(suite),
             'tool_calls':0,'model_calls':0}
        with self.lock: self.state['agent_run']=run;self.state['comparison']=None
        for i,case in enumerate(cases):
            self.update_job('agent',f"Running the agent on {case['id']}…",i,len(cases))
            config={**agent,'_case':case,
                    '_storage_dir':str(self.directory/'agent-environments'/uuid.uuid4().hex)}
            metadata,response=run_agent_case(case['input'],config)
            row={'case_id':case['id'],'input':case['input'],'expected_output':case['expected_output'],
                 'agent_output':metadata['recommendation'],'agent_metadata':metadata,'usage':response.get('usage'),
                 'cost_usd':estimate_cost(response.get('usage'),self.pricing.get('agent',self.pricing['llm'])),
                 'trace':response.get('trace',[]),'turn_outputs':response.get('turn_outputs',[]),
                 'tool_state':response.get('tool_state'),'checks':check_execution(case,response),
                 'scenario_id':case.get('scenario_id')}
            with self.lock:
                run['rows'].append(row)
                run['tool_calls']+=metadata['tool_calls']
                run['model_calls']+=sum(e.get('type')=='model_call' for e in row['trace'])
                run['wall_ms']=round((time.perf_counter()-started)*1000)
                for k in ['input_tokens','output_tokens']:
                    v=(response.get('usage') or {}).get(k)
                    run['usage'][k]=(run['usage'][k]+v) if isinstance(v,int) and run['usage'][k] is not None else None
                run['cost_usd']=(run['cost_usd']+row['cost_usd']) if run['cost_usd'] is not None and row['cost_usd'] is not None else None
        run['status']='complete'
        (self.directory/'agent_outputs.jsonl').write_text(''.join(dumps(r)+'\n' for r in run['rows']))
        self.update_job('agent','Agent outputs are saved. Both judges will receive these exact answers.',len(cases),len(cases))

    def compare(self):
        suite=deepcopy(self.state['suite']);agent=deepcopy(self.state['agent']);agent_run=deepcopy(self.state['agent_run'])
        if agent_run['suite_hash']!=digest(suite): raise ValueError('Eval set changed. Run the agent again.')
        rubric=suite['rubric']; run_id=now().replace(':','').replace('+','-')+'-'+uuid.uuid4().hex[:6]
        comparison={'run_id':run_id,'status':'running','started_at':now(),'suite_hash':digest(suite),
                    'agent_outputs_hash':digest(agent_run['rows']),'pricing':deepcopy(self.pricing),
                    'rubric':deepcopy(rubric),'agent':deepcopy(agent),
                    'agent_run':{k:deepcopy(v) for k,v in agent_run.items() if k!='rows'},
                    'generation':{k:deepcopy(v) for k,v in suite.items() if k not in ('rubric','test_cases')},
                    'rows':[{**r,'judges':{}} for r in agent_run['rows']],'summary':{}}
        comparison['evaluation_config']=evaluation_config(rubric, {'jev': self.pricing['jev']['model'], 'llm': self.pricing['llm']['model']}, [
            {'path': 'context', 'description': 'Agent domain context', 'content': agent['knowledge']},
            {'path': 'rubric', 'description': 'Generated scoring dimensions and levels', 'content': rubric},
            {'path': 'test_cases', 'description': 'Generated cases and execution checks', 'content': suite['test_cases']}],llm_options=self.pricing['llm'])
        with self.lock:self.state['comparison']=comparison
        wall={'jev':0.0,'llm':0.0};total=len(comparison['rows'])*2;count=0
        # Alternate backend order by case to reduce consistent first/second bias.
        for i,row in enumerate(comparison['rows']):
            state={'input':row['input'],'context':agent['knowledge'],'agent_output':row['agent_output'],'expected_output':row['expected_output'],
                   'tool_trace':row.get('trace',[]),'session_outputs':row.get('turn_outputs',[]),
                   'independent_checks':row.get('checks',[]),'final_tool_state':row.get('tool_state')}
            row['judge_requests']=judge_requests(state,rubric,{'jev':self.pricing['jev']['model'],'llm':self.pricing['llm']['model']},llm_options=self.pricing['llm'])
            for backend in (('jev','llm') if i%2==0 else ('llm','jev')):
                model=self.pricing[backend]['model']
                self.update_job('compare',f"{backend.upper()} evaluating {row['case_id']}…",count,total)
                start=time.perf_counter()
                if row['agent_metadata'].get('error'):
                    judge={'backend':backend,'error':'agent_failed','latency_ms':0,'model_id':model,
                           'usage':{'input_tokens':0,'output_tokens':0}}
                elif backend=='jev':
                    judge=evaluate_jev(deepcopy(state),deepcopy(rubric),model_id=model,max_attempts=1)
                elif self.pricing['llm'].get('provider')=='bedrock':
                    judge=evaluate_bedrock(deepcopy(state),deepcopy(rubric),model_id=model,region=self.pricing['llm']['region'],max_tokens=self.pricing['llm'].get('max_tokens',8192),effort=self.pricing['llm'].get('effort','high'))
                else: judge=evaluate_llm(deepcopy(state),deepcopy(rubric),model_id=model)
                wall[backend]+=(time.perf_counter()-start)*1000
                judge['cost_usd']=estimate_cost(judge.get('usage'),self.pricing[backend])
                if 'error' not in judge:
                    validated=validate_judge(judge,rubric)
                    if 'error' in validated:
                        validated.update({k:judge[k] for k in ('usage','latency_ms','cost_usd','model_id','resolved_model_id') if k in judge})
                    judge=validated
                if 'error' not in judge:
                    judge['weighted_score']=sum(d['weight']*judge['scores'][d['name']]/d['scale'] for d in rubric['dimensions'])
                    judge['rubric_passed']=judge['weighted_score']>=rubric['pass_threshold']
                    judge['checks_passed']=all(c['passed'] for c in row.get('checks',[]))
                    judge['passed']=judge['rubric_passed'] and judge['checks_passed']
                with self.lock:
                    row['judges'][backend]=judge
                    comparison['summary']=self.summarize_comparison(comparison,wall)
                count+=1
                self.save()
        comparison['status']='complete';comparison['finished_at']=now()
        comparison['summary']=self.summarize_comparison(comparison,wall)
        (self.directory/(run_id+'.json')).write_text(dumps(comparison))
        with self.lock:
            self.state['history'].insert(0,{'run_id':run_id,'started_at':comparison['started_at'],'summary':comparison['summary']})
        self.update_job('compare','Both judges evaluated the same saved outputs. Comparison ready.',total,total)

    def summarize_comparison(self, comparison, wall):
        result={};n=len(comparison['rows'])
        for backend in ['jev','llm']:
            judges=[r['judges'][backend] for r in comparison['rows'] if backend in r['judges']]
            valid=[j for j in judges if 'error' not in j]
            latency=[j['latency_ms'] for j in judges if isinstance(j.get('latency_ms'),(int,float))]
            costs=[j.get('cost_usd') for j in judges]
            def tokens(key):
                values=[(j.get('usage') or {}).get(key) for j in judges]
                return sum(values) if values and all(type(v)is int for v in values) else None
            result[backend]={'n_scored':len(valid),'n_total':n,'n_completed':len(judges),'n_failed':len(judges)-len(valid),
                'mean_latency_ms':statistics.mean(latency) if latency else None,'p50_latency_ms':statistics.median(latency) if latency else None,
                'wall_ms':round(wall[backend]),'cost_usd':sum(costs) if costs and all(c is not None for c in costs) else None,
                'input_tokens':tokens('input_tokens'),'output_tokens':tokens('output_tokens'),
                'weighted_score':statistics.mean(j['weighted_score'] for j in valid) if valid else None,
                'pass_rate':sum(j['passed'] for j in valid)/n if n else None,
                'model':comparison['pricing'][backend]['model']}
        complete=all(result[b]['n_completed']==n and result[b]['n_failed']==0 for b in ['jev','llm'])
        j,l=result['jev'],result['llm']
        result['speedup']=(l['wall_ms']/j['wall_ms']) if complete and j['wall_ms'] else None
        result['cost_savings_pct']=(1-j['cost_usd']/l['cost_usd'])*100 if complete and l['cost_usd'] and j['cost_usd'] is not None else None
        pairs=[r for r in comparison['rows'] if all('weighted_score' in r['judges'].get(b,{}) for b in ['jev','llm'])]
        result['mean_score_gap']=statistics.mean(abs(r['judges']['jev']['weighted_score']-r['judges']['llm']['weighted_score']) for r in pairs) if pairs else None
        result['paired_n']=len(pairs)
        result['verdict_disagreements']=sum(r['judges']['jev']['passed']!=r['judges']['llm']['passed'] for r in pairs)
        return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def send_json(self,data,status=200):
        content=dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/state': return self.send_json(self.server.app.snapshot())
        if path=='/api/export': return self.send_json(self.server.app.snapshot())
        if path.startswith('/api/runs/'):
            name=path.split('/')[-1]
            if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._' for c in name): return self.send_json({'error':'Invalid run ID'},400)
            file=self.server.app.directory/(name+'.json')
            if not file.exists():return self.send_json({'error':'Run not found'},404)
            return self.send_json(json.loads(file.read_text()))
        if path not in ('/','/index.html'):return self.send_json({'error':'Not found'},404)
        content=(ROOT/'demo/ui/index.html').read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and urlparse(origin).netloc!=self.headers.get('Host'):return self.send_json({'error':'Origin rejected'},403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if length>150000:raise ValueError('Request too large')
            body=json.loads(self.rfile.read(length) or b'{}')
            path=urlparse(self.path).path
            if path=='/api/agent':self.server.app.configure(body)
            else:self.server.app.launch(path.removeprefix('/api/'))
            self.send_json({'ok':True},202)
        except (ValueError,TypeError) as exc:self.send_json({'error':str(exc)},400)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8766);parser.add_argument('--data-dir',type=Path,default=ROOT/'artifacts/tool-memory-demo');parser.add_argument('--model',default=os.environ.get('DEMO_LLM_MODEL','openai.gpt-oss-120b'))
    args=parser.parse_args();pricing=json.loads((ROOT/'demo/pricing.json').read_text())
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler);server.app=Workbench(args.data_dir,pricing,args.model)
    print(f'Eval workbench ready at http://127.0.0.1:{args.port}',flush=True);server.serve_forever()

if __name__=='__main__':main()
