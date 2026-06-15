from config import client, MODEL_NAME
from rag import build_vector_store, run_rag_search
from context import compact_history, should_compact

conversation_history: list[dict] = [] # raw verbatim tail
compacted_state_summary: str = "No previous history compressed yet." # everything older lives here


def chat_with_agent(user_message: str) -> None:
    global conversation_history, compacted_state_summary

    rag_context = run_rag_search(user_message) # pull the most relevant code chunks for this question

    full_user_payload = ( # bundle the user message and retrieved context into one payload
        f"User Request: {user_message}\n\n[Retrieved RAG Context]:\n{rag_context}"
    )
    conversation_history.append({"role": "user", "content": full_user_payload})

    if should_compact(conversation_history):  # squash old history down if we're close to the token limit
        conversation_history, compacted_state_summary = compact_history(
            conversation_history, compacted_state_summary
        )

    system_prompt = ( # inject the running summary so the AI never fully forgets earlier turns
        "You are an AI software auditor. Help the user find bugs and security issues in their codebase.\n"
        f"CRITICAL COMPACTED HISTORICAL STATE: {compacted_state_summary}"
    )
    final_messages = [{"role": "system", "content": system_prompt}] + conversation_history

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=final_messages,
        temperature=0.2,
        stream=True,
    )

    print("\n[AI]: ", end="", flush=True)
    assistant_response = ""
    for chunk in response:
        delta = chunk.choices[0].delta.content or ""
        print(delta, end="", flush=True)
        assistant_response += delta
    print("\n" + "-" * 50)
    conversation_history.append({"role": "assistant", "content": assistant_response})


if __name__ == "__main__":
    build_vector_store()
    print("Ready. Ask anything about the codebase, or type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        chat_with_agent(user_input)