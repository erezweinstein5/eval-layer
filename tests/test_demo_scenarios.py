"""Independent behavior contracts for the tool/memory demo; no model calls."""
import unittest

from demo.scenarios import attach_contracts, check_execution, scenario_contracts


def generated():
    return [{"id": str(i), "scenario_id": c["id"], "input": c["task"],
             "expected_output": "Unverified generator text",
             "metadata": {"difficulty": c["difficulty"]}}
            for i, c in enumerate(scenario_contracts())]


def call(name, arguments=None, result=None, turn=1):
    return {"type": "tool_call", "name": name, "arguments": arguments or {},
            "result": result or {"ok": True}, "turn": turn}


class ScenarioTests(unittest.TestCase):
    def case(self, name):
        return next(c for c in attach_contracts(generated()) if c["scenario_id"] == name)

    def test_unverified_reference_is_replaced_by_known_contract(self):
        case = self.case("old-refund")
        self.assertIn("window has expired", case["expected_output"])
        self.assertEqual(case["workspace_id"], "ws-acme")
        self.assertNotIn("Unverified", case["expected_output"])

    def test_missing_or_duplicate_scenario_is_rejected(self):
        cases = generated()
        cases[-1]["scenario_id"] = cases[0]["scenario_id"]
        with self.assertRaises(ValueError):
            attach_contracts(cases)

    def test_memory_claim_without_execution_cannot_pass(self):
        response = {"text": "I saved it and remembered in_app.", "trace": []}
        checks = check_execution(self.case("remember-preference"), response)
        self.assertFalse(all(c["passed"] for c in checks))

    def test_memory_must_be_retrieved_in_fresh_session(self):
        trace = [
            call("save_memory", {"key": "preferred_channel", "value": "in_app"}),
            call("search_memory", {"query": "preferred channel"},
                 {"memories": [{"key": "preferred_channel", "value": "in_app"}]}, turn=1),
        ]
        response = {"trace": trace, "tool_state": {"memory": {"preferred_channel": {"value": "in_app"}}}}
        checks = check_execution(self.case("remember-preference"), response)
        self.assertFalse(next(c["passed"] for c in checks if "Fresh session" in c["name"]))
        trace[-1]["turn"] = 2
        checks = check_execution(self.case("remember-preference"), response)
        self.assertTrue(all(c["passed"] for c in checks))

    def test_failed_invoice_lookup_does_not_count_as_evidence(self):
        checks = check_execution(self.case("recent-refund"), {"trace": [
            call("get_invoice", {"invoice_id": "inv-200"}, {"error": "not_found"}),
            call("search_policy"),
        ]})
        self.assertFalse(next(c["passed"] for c in checks if c["name"] == "Read inv-200"))

    def test_unauthorized_mutation_attempt_fails_even_if_tool_denies_it(self):
        checks = check_execution(self.case("untrusted-memory"), {"trace": [
            call("search_memory"), call("create_ticket", result={"error": "denied"}),
        ]})
        self.assertFalse(next(c["passed"] for c in checks if "No unauthorized create_ticket" == c["name"]))

    def test_tenant_refusal_and_denied_lookup_are_valid(self):
        for trace in ([], [call("get_invoice", {"invoice_id": "inv-beta"}, {"error": "not_found"})]):
            self.assertTrue(all(c["passed"] for c in check_execution(self.case("tenant-isolation"), {"trace": trace})))
        leaked = [call("get_invoice", {"invoice_id": "inv-beta"}, {"amount": 999})]
        self.assertFalse(all(c["passed"] for c in check_execution(self.case("tenant-isolation"), {"trace": leaked})))

    def test_ticket_requires_exactly_one_successful_creation(self):
        ticket = {"ticket_id": "ticket-001", "category": "billing"}
        trace = [call("search_memory"),
                 call("get_invoice", {"invoice_id": "inv-100"}),
                 call("get_invoice", {"invoice_id": "inv-101"}),
                 call("create_ticket", {"category": "billing"}, {"ticket": ticket, "created": True})]
        response = {"trace": trace, "tool_state": {"tickets": {"ticket-001": ticket}}}
        self.assertTrue(all(c["passed"] for c in check_execution(self.case("duplicate-charge"), response)))
        trace.append(call("create_ticket", {"category": "billing"}, {"ticket": ticket, "created": True}))
        self.assertFalse(all(c["passed"] for c in check_execution(self.case("duplicate-charge"), response)))

    def test_ticket_response_without_persisted_record_fails(self):
        ticket = {"ticket_id": "ticket-missing", "category": "billing"}
        trace = [call("search_memory"), call("get_workspace"), call("search_policy"),
                 call("create_ticket", {"category": "billing"}, {"ticket": ticket, "created": True})]
        checks = check_execution(self.case("seat-escalation"), {"trace": trace, "tool_state": {"tickets": {}}})
        self.assertFalse(next(c["passed"] for c in checks if c["name"] == "Billing ticket persisted or verified"))


if __name__ == "__main__":
    unittest.main()
