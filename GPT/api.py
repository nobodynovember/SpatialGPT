from openai import OpenAI
import base64
import os

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


generation_key = ""  # GPT key
client = OpenAI(
    api_key=generation_key,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def gpt_infer(system, text, image_list, model="gpt-4-vision-preview", max_tokens=600, response_format=None, defined_indice=None): 

    user_content = []
    for i, image in enumerate(image_list):
        if image is not None:
            
            if defined_indice:
                j = defined_indice[i]
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {j}:"
                    },
                )
            else:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"Image {i}:"
                    },
                )
            with open(image, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            image_message = {
                     "type": "image_url",
                     "image_url": {
                         "url": f"data:image/jpeg;base64,{image_base64}",
                         "detail": "low"
                     }
                 }
            user_content.append(image_message)

    user_content.append(
        {
            "type": "text",
            "text": text
        }
    )

    messages = [
        {"role": "system",
         "content": system
         },
        {"role": "user",
         "content": user_content
         }
    ]
    # use the same version of 4o model with previous zero-shot model for agent performance comparision
    # can be changed to other cheaper versions of 4o model for cost saving, but with slower api response 
    model="gpt-4o-2024-05-13" 
    if response_format:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens, response_format=response_format)
    else:
        chat_message = completion_with_backoff(model=model, messages=messages, temperature=0, max_tokens=max_tokens)

    #print('gpt chat_message:', chat_message)
    answer = chat_message.choices[0].message.content
    tokens = chat_message.usage

    return answer, tokens


