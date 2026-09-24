import json,os,unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from scripts.bedrock_judge import build_request,evaluate,normalize_usage

RUBRIC={'dimensions':[{'name':'quality','scale':2,'weight':1,'levels':{1:'Wrong',2:'Right'}}],'pass_threshold':0.7}
GRADE={'scores':{'quality':2},'details':{'quality':{'reasoning':'Good','evidence':['Output'],'suggestion':'None','confidence':'high'}},'overall_reasoning':'Good'}
class Stream:
 def __init__(self,response):self.response=response
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def get_final_message(self):return self.response
class Client:
 def __init__(self,stop='end_turn',text=None):self.messages=self;self.stop=stop;self.text=text;self.payload=None
 def stream(self,**payload):
  self.payload=payload
  return Stream(NS(model='claude-opus-5-5',stop_reason=self.stop,usage=NS(model_dump=lambda:{'input_tokens':10,'output_tokens':20,'cache_read_input_tokens':5,'cache_creation_input_tokens':0}),content=[NS(type='thinking',thinking='private'),NS(type='text',text=self.text or json.dumps(GRADE))]))
class BedrockJudgeTests(unittest.TestCase):
 def test_request_and_scoring(self):
  c=Client();r=evaluate({'input':'Test'},RUBRIC,model_id='us.anthropic.claude-opus-5-5',region='us-east-1',client=c)
  self.assertNotIn('error',r);self.assertEqual(r['scores']['quality'],2)
  self.assertEqual(r['usage']['input_tokens'],15);self.assertEqual(r['resolved_model_id'],'claude-opus-5-5')
  self.assertEqual(c.payload['output_config'],{'effort':'high'});self.assertNotIn('temperature',c.payload)
  self.assertNotIn('private',str(r));self.assertNotIn('raw_response',r)
 def test_truncation_is_failure_with_usage(self):
  r=evaluate({},RUBRIC,model_id='opus',region='us-east-1',client=Client(stop='max_tokens'))
  self.assertEqual(r['error'],'incomplete_response');self.assertEqual(r['usage']['output_tokens'],20)
 def test_configuration_fails_before_transport(self):
  c=Client();r=evaluate({},RUBRIC,model_id='opus',region='evil.example',client=c)
  self.assertEqual(r['error'],'configuration_error');self.assertIsNone(c.payload)
 def test_parse_failure_preserves_metadata(self):
  r=evaluate({},RUBRIC,model_id='opus',region='us-east-1',client=Client(text='bad json'))
  self.assertIn('error',r);self.assertEqual(r['usage']['input_tokens'],15)
 def test_secret_echo_is_redacted(self):
  with patch.dict(os.environ,{'AWS_BEARER_TOKEN_BEDROCK':'secret-value'}):
   r=evaluate({},RUBRIC,model_id='opus',region='us-east-1',client=Client(text=json.dumps(GRADE).replace('Good','secret-value')))
  self.assertNotIn('secret-value',str(r))
 def test_unpriced_cache_write_cost_is_unknown(self):
  from scripts.llm_judge import estimate_cost
  self.assertIsNone(estimate_cost({"input_tokens":100,"output_tokens":10,"cache_creation_input_tokens":50},{"input_usd_per_million":4.4,"output_usd_per_million":22}))
 def test_unknown_usage(self):
  self.assertNotIn('input_tokens',normalize_usage({'output_tokens':3}))
  self.assertNotIn('input_tokens',normalize_usage({'input_tokens':True,'output_tokens':3}))
if __name__=='__main__':unittest.main()
