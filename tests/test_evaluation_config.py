import unittest
from scripts.evaluation_config import evaluation_config, judge_requests
from scripts.llm_judge import build_messages
from scripts.jev_judge import build_request

class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.rubric={'dimensions':[{'name':'correctness','weight':1,'scale':2,'levels':{1:'Wrong',2:'Correct'}}],'pass_threshold':0.8}
        self.models={'jev':'jev-1.13.0','llm':'test-judge'}

    def test_payloads_match_adapters_and_preserve_evidence(self):
        evidence={'input':'question','trace':[{'name':'search_memory','result':{'value':'he'}}]}
        requests=judge_requests(evidence,self.rubric,self.models)
        self.assertEqual(requests['jev'],build_request(evidence,self.rubric,self.models['jev']))
        self.assertEqual(requests['llm']['messages'],build_messages(evidence,self.rubric))
        self.assertNotIn('Authorization',str(requests))
        self.assertEqual(requests['llm']['temperature'],0)

    def test_config_freezes_artifacts_and_model_identity(self):
        artifacts=[{'path':'context.md','content':{'source':'original'}}]
        config=evaluation_config(self.rubric,self.models,artifacts,provenance='reconstructed')
        artifacts[0]['content']['source']='changed'
        self.models['llm']='changed'
        self.assertEqual(config['artifacts'][0]['content']['source'],'original')
        self.assertEqual(config['backends']['llm']['model'],'test-judge')
        self.assertEqual(config['provenance'],'reconstructed')

if __name__=='__main__':unittest.main()
