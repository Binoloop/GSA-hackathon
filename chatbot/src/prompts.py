

class TCprompts:
    def __init__(self, content, question):
        """
        Class for generating prompts used in context and name extraction tasks.

        Attributes:
        - content: The content used to generate the prompt.

        Methods:
        - context_prompt(): Generates a context prompt for a given content.
        - name_prompt(): Generates a prompt for extracting names from a given text.
        """
        self.content = content
        self.question = question

    def context_prompt(self):
        """
        Generates a context prompt based on user question.

        Returns:
        - str: A formatted context prompt based on the content.
        """
        try:
            add_context = """"
                        You are a knowledgeable assistant. Use the following context to answer the question.

                        Context: {context}

                        Question: {question}
                        Answer:
                        """.format(context=self.content, question=self.question)

            return add_context
        except Exception as e:
            print(f"Error while processing context prompt: {str(e)}")
