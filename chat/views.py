import json
import traceback
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

from gpt.inference import generate_response
from gpt.loader import get_model, get_tokenizer, get_device_str


def index(request):
    if not request.session.session_key:
        request.session.save()
    return render(request, "chat/index.html")


@csrf_exempt
@require_http_methods(["POST"])
def chat_api(request):
    try:
        body     = json.loads(request.body.decode("utf-8"))
        user_msg = body.get("message", "").strip()

        if not user_msg:
            return JsonResponse({"error": "Message cannot be empty."}, status=400)

        history = request.session.get("conversation_history", []) or []
        history.append({"role": "user", "content": user_msg})

        model     = get_model()
        tokenizer = get_tokenizer()
        device    = get_device_str()

        reply = generate_response(
            model=model,
            tokenizer=tokenizer,
            conversation_history=history,
            device=device,
            max_new_tokens=300,
            temperature=0.7,
            top_k=40,
        )

        # If empty — return early WITHOUT saving to history (avoids poisoning future prompts)
        if not reply or not reply.strip():
            return JsonResponse({
                "reply": "I couldn't generate a response. Please try again.",
                "history_length": len(history),
            })

        # Only reach here if reply is valid — safe to save to history
        history.append({"role": "assistant", "content": reply})
        if len(history) > 10:
            history = history[-10:]

        request.session["conversation_history"] = history
        request.session.modified = True

        return JsonResponse({"reply": reply, "history_length": len(history)})

    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body."}, status=400)
    except Exception as e:
        print(f"[ChatAPI] Error: {e}")
        traceback.print_exc()
        return JsonResponse({"error": "An internal error occurred."}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def reset_conversation(request):
    request.session["conversation_history"] = []
    request.session.modified = True
    return JsonResponse({"status": "ok", "message": "Conversation cleared."})