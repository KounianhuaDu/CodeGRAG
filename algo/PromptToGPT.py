import openai
import logging
logging.basicConfig(format='%(levelname)s %(asctime)s %(process)d %(message)s', level=logging.INFO)

def generate_response(prompt, model="gpt-4"):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    message = response.choices[0]["message"]["content"]
    return message.strip()

def api_test(id, prompt,model_name):
    try:
        response = generate_response(prompt,model_name)
        return response
    except:
        logging.info("Problem occurred with %s using %s api"%(str(id),model_name))
        return None
