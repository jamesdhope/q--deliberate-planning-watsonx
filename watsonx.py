from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watsonx_ai import APIClient, Credentials
import os
import numpy as np

# To display example params enter
GenParams().get_example_values()

generate_params = {
    GenParams.MAX_NEW_TOKENS: 250,
    GenParams.RETURN_OPTIONS: {'token_logprobs': True, 'input_text': False, 'generated_tokens': True, 'input_tokens': True}
}

watsonx_api_key = os.environ['WATSONX_APIKEY']
watsonx_project_id = os.environ['PROJECT_ID']

expert_model = Model(
    model_id="meta-llama/llama-3-70b-instruct",
    params=generate_params,
    credentials=Credentials(
                    api_key = f"{watsonx_api_key}",
                    url = "https://eu-gb.ml.cloud.ibm.com"),
                    project_id = f"{watsonx_project_id}"
    )

primary_model = Model(
    model_id="meta-llama/llama-3-70b-instruct",
    params=generate_params,
    credentials=Credentials(
                    api_key = f"{watsonx_api_key}",
                    url = "https://eu-gb.ml.cloud.ibm.com"),
                    project_id = f"{watsonx_project_id}"
    )


