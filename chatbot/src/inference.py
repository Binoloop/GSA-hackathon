
import google.api_core.exceptions as google_exceptions
import openai
import requests
from constants import LLM_PARAMETER_TEMPERATURE, INITIAL_BACKOFF, MAX_RETRIES, BACKOFF_MULTIPLIER, NON_CONSTANT_BACKOFF_VALUES, CONSTANT_BACKOFF, BACKOFF_JITTER
import backoff  # for exponential backoff
import numpy as np
import os
import random
import time
from dotenv import load_dotenv
load_dotenv()  # Load variables from .env file


def retry_with_exponential_backoff(
    func,
    jitter: bool = BACKOFF_JITTER,
    max_retries: int = MAX_RETRIES,
    constant_val: int = CONSTANT_BACKOFF,
    initial_value: int = INITIAL_BACKOFF,
    base_value: int = BACKOFF_MULTIPLIER,
    non_constant_values: int = NON_CONSTANT_BACKOFF_VALUES,
    errors: tuple = (openai.APIError,
                     openai.RateLimitError,
                     openai.InternalServerError,
                     openai.APIConnectionError,
                     openai.APIStatusError,
                     openai.APITimeoutError,
                     openai.AuthenticationError,
                     ),
):
    """Retry a function with exponential backoff."""

    # Debugging code to check error types
    for error in errors:
        if not issubclass(error, BaseException):
            raise TypeError(f"Provided error {error.__name__} does not inherit from BaseException")

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0

        # Creating a list of possible delays
        delay_arr = [initial_value * base_value ** i for i in range(non_constant_values)] + [constant_val] * (max_retries - non_constant_values)

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                original_delay_value = delay_arr[num_retries - 1]

                # Delaying such that the delay is increased by adding random jitter
                delay = float(original_delay_value + (jitter * random.uniform(0, original_delay_value)))
                print("Delayed by: {}".format(delay))

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print(f"Unexpected error occurred: {str(e)}", exc_info=True)
                # raise e

    return wrapper


class DAModelInference():

    def __init__(self):
        pass

    @retry_with_exponential_backoff
    def _openai(self,
                prompt: str,
                model_name: str = 'gpt-3.5-turbo',
                temperature: float = LLM_PARAMETER_TEMPERATURE,
                messages: list = None) -> str:
        """
        Using  OpenAI API to get answer to the prompts

        Parameters
        ----------
        prompt : str
            The prompt that will be sent to the LLM using api
        model_name : str
            The name of the model that was used to perform text generation

        Returns
        -------
        response : str
            The answer to the prompt by the LLM
        """
        API_KEY = os.getenv('OPENAI_API_KEY')

        client = openai.OpenAI(api_key=API_KEY)

        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        else:
            messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            timeout=60)

        text_response = response.choices[0].message.content
        input_token_count = response.usage.prompt_tokens
        output_token_count = response.usage.completion_tokens

        return text_response, input_token_count, output_token_count

    # Creating code to get responses from PaLM2 model.

    # from constants import LLM_PARAMETER_TEMPERATURE
    def _check_and_assign_inference_model(self, model_service_api):
        # Select function for self.model_inference based on the model_service_api
        if model_service_api == "openai":
            model_inference = self._openai
        else:
            raise ValueError("Invalid model service API. Please choose 'openai', 'togetherai', 'google' or 'huggingface'.")

        return model_inference