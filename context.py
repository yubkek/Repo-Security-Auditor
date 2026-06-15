import tiktoken
from config import client, MODEL_NAME, TOKEN_THRESHOLD

tokenizer = tiktoken.encoding_for_model(MODEL_NAME)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def compact_history(
    conversation_history: list[dict],
    compacted_state_summary: str,
) -> tuple[list[dict], str]:
    print("\n[COMPACTOR] Token threshold breached, Compacting History")

    tail_history = conversation_history[-2:] # keep the freshest turns verbatim
    head_history = conversation_history[:-2] # everything older gets squashed

    if not head_history:
        print("[COMPACTOR] Nothing to compact yet.") # happens on the very first trigger
        return conversation_history, compacted_state_summary

    history_string = "".join(
        f"{msg['role'].upper()}: {msg['content']}\n" for msg in head_history
    )

    compaction_prompt = (
        "You are a Context Compactor backend worker. Compress the following software "
        "engineering chat history into a dense, brief summary. Focus strictly on "
        "what files were looked at, what bugs were identified, and what actions failed/succeeded. "
        f"Incorporate past context if relevant: {compacted_state_summary}\n\n"
        f"History to compress:\n{history_string}"
    )

    response = client.chat.completions.create( # cheap background call - temperature 0 so it stays factual
        model=MODEL_NAME,
        messages=[{"role": "user", "content": compaction_prompt}],
        temperature=0.0,
    )

    new_summary = response.choices[0].message.content
    print("[COMPACTOR] Done. Raw history wiped. Summary updated")
    return tail_history, new_summary

def should_compact(conversation_history: list[dict]) -> bool:
    raw_text = " ".join(m["content"] for m in conversation_history)
    total = count_tokens(raw_text)
    print(f"[SYSTEM] Active History Tokens: {total} / {TOKEN_THRESHOLD}")
    return total > TOKEN_THRESHOLD  # caller decides what to do when this is True
