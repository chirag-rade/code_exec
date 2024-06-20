import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests


class LLMasEvaluator:

    def __init__(
        self,
        base_url,
        engagement_name,
        project_name,
        test_auth_on_init=True,
        log_request_payload=False,
    ):
        self.base_url = f"{base_url}/api/v1"
        self.project_name = project_name
        self.engagement_name = engagement_name
        self.log_request_payload = log_request_payload
        self.auth_headers = {
            "Authorization": f'Bearer {os.getenv("LLM_AS_EVALUATOR_API_KEY")}'
        }
        if test_auth_on_init:
            self.test_auth()

    def refresh_token(self):
        """
        Refresh the API token.
        """
        url = f"{self.base_url}/a/refresh-token"
        payload = {
            "client_id": os.getenv("LLM_AS_EVALUATOR_CLIENT_ID"),
            "client_secret": os.getenv("LLM_AS_EVALUATOR_CLIENT_SECRET"),
        }
        if self.log_request_payload:
            print("=" * 90)
            print("@LOG@")
            print(url)
            print(payload)
            print("=" * 90)
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            new_token = response.json().get("access_token")
            self.auth_headers["Authorization"] = f"Bearer {new_token}"
            print("Token refreshed and updated in your env.")
        else:
            print(f"Failed to refresh token with status code: {response.status_code}")
            print(response.text)

    def test_auth(self):
        """
        Test the authentication by making a request to the auth endpoint.
        """
        url = f"{self.base_url}/a/test-user-project-auth"
        payload = {
            "project_name": self.project_name,
            "engagement_name": self.engagement_name,
            "email": "current_user",
        }
        response = self._make_request("POST", url, json=payload)
        if response:
            print("Authentication successful.")
        else:
            print("Authentication failed.")

    def prepare_single_payload_no_eval(
        self,
        batch_name,
        inputs,
        evaluations,
        aggregated_evaluations=[],
        parse=True,
        format_to_issues_scores=False,
        is_dev_request=False,
        **item_metadata,
    ):
        return {
            "batch_name": batch_name,
            "engagement_name": self.engagement_name,
            "project_name": self.project_name,
            "item_metadata": item_metadata,
            "input_type": "parsed_json_args",
            "inputs": inputs,
            "evaluations": evaluations,
            "aggregated_evaluations": aggregated_evaluations,
            "parse": parse,
            "format_to_issues_scores": format_to_issues_scores,
            "is_dev_request": is_dev_request,
        }

    def _make_request(self, method, url, **kwargs):
        """
        Make an HTTP request and handle token expiration.
        """
        if self.log_request_payload:
            print("=" * 90)
            print("@LOG@")
            print(url)
            if "json" in kwargs:
                print(kwargs["json"])
            print("=" * 90)
        response = requests.request(method, url, headers=self.auth_headers, **kwargs)
        if (
            response.status_code == 401
        ):  # Unauthorized, possibly due to token expiration
            self.refresh_token()
            response = requests.request(
                method, url, headers=self.auth_headers, **kwargs
            )

        if response.status_code == 200:
            return response.json()
        else:
            print(response.text)
            return None

    def make_post_request(self, payload):
        url = f"{self.base_url}/runs/"
        return self._make_request("POST", url, json=payload)

    def parse_notebooks(self, notebooks_content):
        """
        Make a POST request with the parsed content.
        """

        def parse_content(notebook_content):
            url = f"{self.base_url}/helpers/inputs/parse-only"
            return self._make_request(
                "POST", url, json={"downloaded_content": notebook_content}
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(parse_content, notebooks_content))
        return [result.get("parsed_content", {}) for result in results]

    def list_evaluator_types(self):
        """
        List available evaluator types.
        """
        url = f"{self.base_url}/e/evaluator-types/"
        return self._make_request("GET", url)

    def upsert_evaluator_by_name_from_override_config(
        self, override_config, description=None, override_name=None
    ):
        payload = dict(override_config)
        if override_name is not None:
            payload["name"] = override_name
        if description is not None:
            payload["description"] = description
        llm_config = payload.pop("llm_config", {})
        for k, v in llm_config.items():
            payload["llm_" + k] = v
        return self.upsert_evaluator_by_name(payload)

    def upsert_evaluator_by_name(self, evaluator_data):
        """
        Upsert an evaluator by name.
        """
        if not evaluator_data.get("name"):
            print("Error: 'name' is required in evaluator_data.")
            return None

        # if not evaluator_data.get("evaluator_type_id") and not evaluator_data.get(
        #     "evaluator_type_name"
        # ):
        #     print(
        #         "Error: Either 'evaluator_type_id' or 'evaluator_type_name' is required in evaluator_data."
        #     )
        #     return None

        url = f"{self.base_url}/e/evaluators/upsert-by-name"
        return self._make_request("POST", url, json=evaluator_data)

    def list_evaluators(self):
        """
        Fetches a list of evaluators from the API.
        """
        url = f"{self.base_url}/e/evaluators/"
        return self._make_request("GET", url)

    def get_evaluator_by_name(self, name):
        """
        Fetches an evaluator by name from the API.
        """
        encoded_name = urllib.parse.quote(name)
        url = f"{self.base_url}/e/evaluators/{encoded_name}"
        return self._make_request("GET", url)

    def init_evaluator_config_override(self):
        return {
            "evaluator_type_name": None,
            "name": None,
            "config": None,
            "llm_config": {
                "provider": None,
                "model": None,
                "params": None,
            },
            "input_schema": None,
            "output_schema": None,
        }

    def create_evaluation_config(
        self,
        evaluator_name,
        id_name,
        use_for_agg_layer,
        config={},
        evaluator_config_override=None,
    ):
        evaluation_config = {
            "evaluator_name": evaluator_name,
            "name": id_name,
            "use_for_agg_layer": use_for_agg_layer,
            "config": config,
        }
        if evaluator_config_override is not None:
            evaluation_config["evaluator_config_override"] = evaluator_config_override
        return evaluation_config

    def parallel_calling_bulk(
        self,
        batch_name,
        inputs,
        evaluations,
        aggregated_evaluations=[],
        parse=True,
        format_to_issues_scores=False,
        is_dev_request=False,
        **item_metadata,
    ):
        """
        Initiates batch run for each input in parallel.
        """

        def worker(single_input):
            return self.initiate_batch_run(
                batch_name,
                [single_input],
                evaluations,
                aggregated_evaluations=aggregated_evaluations,
                parse=parse,
                format_to_issues_scores=format_to_issues_scores,
                is_dev_request=is_dev_request,
                **item_metadata,
            )

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(worker, inputs))

        return results

    def initiate_batch_run(
        self,
        batch_name,
        inputs,
        evaluations,
        aggregated_evaluations=[],
        parse=True,
        format_to_issues_scores=False,
        is_dev_request=False,
        **item_metadata,
    ):
        """
        Create payload and make a POST request.
        """
        if len(inputs) > 10:
            raise ValueError(
                f"The number of inputs exceeds the allowed limit of 10. Was: {len(inputs)}"
            )
        payload = self.prepare_single_payload_no_eval(
            batch_name,
            inputs,
            evaluations,
            aggregated_evaluations=aggregated_evaluations,
            parse=parse,
            format_to_issues_scores=format_to_issues_scores,
            is_dev_request=is_dev_request,
            **item_metadata,
        )
        return self.make_post_request(payload)

    def get_batch_run_status(self, batch_run_id):
        url = f"{self.base_url}/runs/batch/{batch_run_id}/status"
        return self._make_request("GET", url)

    def check_batch_status_until_ready(self, batch_run_id, interval=5):
        start_time = datetime.now()
        while True:
            status_response = self.get_batch_run_status(batch_run_id)
            elapsed = f"{datetime.now() - start_time}".split(".")[0]
            runs_left = -1
            runs_total = -1
            if status_response:
                runs_total = len(status_response["runs"])
                runs_left = status_response["runs_left"]
                if status_response.get("is_batch_ready", False):
                    print(f"Batch is ready. Time elapsed: {elapsed}")
                    return status_response
            else:
                raise Exception("Failed to get batch run status")
            print(
                f"Batch not ready, checking again in {interval} seconds. Time elapsed: {elapsed}"
            )
            print(f"Runs left {runs_left}/{runs_total}")
            time.sleep(interval)

    def get_batch_run(self, batch_run_id, check_interval, wait_for_status=True):
        if wait_for_status:
            self.check_batch_status_until_ready(batch_run_id, check_interval)
        url = f"{self.base_url}/runs/batch/{batch_run_id}"
        return self._make_request("GET", url)

    def get_run(self, run_id, check_interval, wait_for_status=True):
        if wait_for_status:
            r = self.check_run_status_until_ready(run_id, check_interval)
        url = f"{self.base_url}/runs/{run_id}"
        return self._make_request("GET", url)

    def get_run_status(self, run_id):
        url = f"{self.base_url}/runs/{run_id}/status"
        return self._make_request("GET", url)

    def check_run_status_until_ready(self, run_id, interval=5):
        start_time = datetime.now()
        while True:
            status_response = self.get_run_status(run_id)
            elapsed = f"{datetime.now() - start_time}".split(".")[0]
            if status_response:
                if status_response.get("status") not in ["pending", "in_progress"]:
                    print(f"Run is ready. Time elapsed: {elapsed}")
                    return status_response
            else:
                raise Exception("Failed to get run status")
            print(
                f"Run not ready, checking again in {interval} seconds. Time elapsed: {elapsed}"
            )
            time.sleep(interval)
