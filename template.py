import jinja2
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Template:
    def __init__(self, generator):
        self.generator = generator
        self.data = {}
        self.max_seq_len = self.generator.model.max_seq_len

    def generate(template):
        pass


class ChatTemplate(Template):
    def __init__(self, generator, chat_template):
        super().__init__(generator)
        # self.chat_template = chat_template
        self.chat_template = "{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"
        self.conversation = []

    def user_input(self, prompt):
        self.conversation.append({"role": "user", "content": prompt})
