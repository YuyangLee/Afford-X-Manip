from helpers.helper import Helper
from openai import OpenAI
import os


class GPTHelper(Helper):
    def __init__(self):
        assert os.environ.get("OPENAI_KEY", None) is not None, "OPENAI_KEY not found in environment variables!"
        self.clear()

    def clear(self):
        self.client = OpenAI()

    def query(self, prompt, system_content="You are a helpful assistant.", terminate_condition=lambda x: len(x) > 0):
        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system_content}, {"role": "user", "content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=4096,
            stream=False,
        )

        return completion.choices[0].message.content
