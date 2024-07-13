import os
import openai

def get_response(model, messages, temperature, max_tokens, current_retry):
  openai.api_key = os.getenv("OPENAI_API_KEY")
  try:
    response = openai.ChatCompletion.create(
      model = model,
      messages = messages,
      temperature = temperature,
      max_tokens = max_tokens
    )
  except openai.error.Timeout as e:
    print(f"OpenAI API request timed out: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.APIError as e:
    print(f"OpenAI API returned an API Error: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.APIConnectionError as e:
    print(f"OpenAI API request failed to connect: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.InvalidRequestError as e:
    print(f"OpenAI API request was invalid: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.AuthenticationError as e:
    print(f"OpenAI API request was not authorized: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.PermissionError as e:
    print(f"OpenAI API request was not permitted: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.RateLimitError as e:
    print(f"OpenAI API request exceeded rate limit: {e}, Retry: {current_retry}")
    response = "error"
  except openai.error.ServiceUnavailableError as e:
    print(f"OpenAI API request was overloaded: {e}, Retry: {current_retry}")
    response = "error"
  return response


def construct_msg(prompt, user_content):
  messages = [
    {
      "role": "system",
      "content": prompt,
    },
    {
      "role": "user",
      "content": user_content
    }
  ]
  return messages


