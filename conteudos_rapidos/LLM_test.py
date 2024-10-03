from LLM import LLM

llm = LLM()

transcripts = [
    "Sejam muito bem-vindos em destaque esta semana à Assembleia Geral das Nações Unidas em Nova Iorque, uma missão militar europeia que acompanhamos em Moçambique.",
    "E ainda, conversamos com os realizadores finalistas do Prémio Europeu de Cinema Luz."
    "Começamos com os principais destaques desta semana. A Assembleia Geral das Nações Unidas reunida em Nova Iorque. O presidente ucraniano foi mais uma vez fazer apelos à paz e avisar para os perigos de um desastre nuclear. Destaca ainda para a França o novo governo reuniu pela primeira vez."
]

title, summary, keywords = llm.process_transcripts(transcripts)
print(f"\nTitle: \n{title}")
print(f"\nSummary: \n{summary}")
print(f"\nKeywords: \n{keywords}")