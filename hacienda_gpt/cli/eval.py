from __future__ import annotations

import json
from statistics import mean

import click
from langchain_core.messages import AIMessage, HumanMessage

from hacienda_gpt.llm.chain import create_openai_chain
from hacienda_gpt.utils import get_openai_api_key


GOLDEN_SET: list[dict[str, str]] = [
    {
        "question": "¿Dónde puedo consultar el calendario del contribuyente?",
        "must_include": "calendario",
    },
    {
        "question": "¿Qué es el IRPF en la Agencia Tributaria?",
        "must_include": "IRPF",
    },
    {
        "question": "¿Dónde encuentro información de aplazamientos y fraccionamientos?",
        "must_include": "aplaz",
    },
]


def _score_answer(answer: str, must_include: str) -> float:
    return 1.0 if must_include.lower() in answer.lower() else 0.0


@click.command()
@click.option("--output", default="./eval_results.json", help="Output json file")
def cli(output: str) -> None:
    chain = create_openai_chain(openai_api_key=get_openai_api_key())
    history: list[HumanMessage | AIMessage] = []
    results = []

    for item in GOLDEN_SET:
        result = chain.invoke({"input": item["question"], "chat_history": history})
        answer = result["answer"]
        score = _score_answer(answer, item["must_include"])
        results.append({"question": item["question"], "answer": answer, "score": score})
        history.extend([HumanMessage(content=item["question"]), AIMessage(content=answer)])

    summary = {"avg_score": mean([entry["score"] for entry in results]), "total": len(results), "results": results}
    with open(output, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli()
