import csv
import time
import random
import datetime
import os
import json
import nltk
from nltk.corpus import words
from openai import OpenAI

# Download NLTK words if necessary
nltk.download('words', quiet=True)
english_words = set(words.words())

# Seed lists for programmatic focus (60% weight)
tech_seeds = ["API", "Database", "Algorithm", "Interface", "Server", "Client", "Network", "Module", "Bot", "Script", "Cloud", "Parser", "Engine", "Framework", "Query", "Widget", "App", "Dashboard", "Cache", "Queue", "Plugin", "Monitor", "Compiler", "Router"]
industry_seeds = ["Productivity", "Health", "Education", "Finance", "Gaming", "Travel", "Social", "Ecommerce", "Media", "Security", "Analytics", "Logistics", "Entertainment", "Communication", "Retail", "HR", "Marketing", "Energy", "Research"]
verb_seeds = ["Tracking", "Filtering", "Displaying", "Processing", "Notifying", "Converting", "Organizing", "Sharing", "Syncing", "Generating", "Analyzing", "Rendering", "Automating", "Integrating", "Streaming", "Fetching", "Updating", "Parsing", "Optimizing", "Logging", "Scheduling", "Pushing", "Pulling"]
noun_seeds = ["Tool", "System", "Report", "Feed", "Alert", "Profile", "List", "Stream", "Dashboard", "Widget", "Engine", "Service", "Data", "Model", "Plugin", "Queue", "Graph", "Log", "Form", "Map", "Index", "Cache", "Flow", "Node"]

# Expand pools with filtered dictionary words
def expand_pool(seed_list, target_size=1000):
    pool = set(seed_list)
    if seed_list == tech_seeds:
        candidates = [w for w in english_words if len(w) > 3 and (w.endswith(("er", "ing", "ion", "or", "et")) or w in tech_seeds) and not w.endswith(("ness", "ity"))]
    elif seed_list == industry_seeds:
        candidates = [w for w in english_words if len(w) > 4 and (w.endswith(("ness", "ity", "ics", "ing")) or w in industry_seeds) and not w.endswith(("er", "or"))]
    elif seed_list == verb_seeds:
        candidates = [w for w in english_words if len(w) > 3 and (w.endswith(("e", "t", "n", "d", "ing")) or w in verb_seeds) and not w.endswith(("ness", "ity"))]
    elif seed_list == noun_seeds:
        candidates = [w for w in english_words if len(w) > 3 and not w.endswith(("ing", "ness", "ity")) and w not in verb_seeds]
    else:
        candidates = english_words
    
    while len(pool) < target_size and candidates:
        word = random.choice(candidates)
        pool.add(word)
        candidates.remove(word)
    return list(pool)

# Generate large pools
technologies = expand_pool(tech_seeds, 1000)
industries = expand_pool(industry_seeds, 1000)
verbs = expand_pool(verb_seeds, 1000)
nouns = expand_pool(noun_seeds, 1000)

# Weighted choice function (60% seed weight)
def weighted_choice(pool, seeds, seed_weight=0.6):
    if random.random() < seed_weight:
        return random.choice(seeds)
    return random.choice(pool)

# Generate a random idea
def generate_random_idea():
    tech = weighted_choice(technologies, tech_seeds)
    industry = weighted_choice(industries, industry_seeds)
    verb = weighted_choice(verbs, verb_seeds)
    noun = weighted_choice(nouns, noun_seeds)
    return f"A {tech} for {verb} {industry} {noun}"

# Crossover two parent ideas
def crossover(parent1, parent2):
    p1_parts = parent1.split()
    p2_parts = parent2.split()
    child_parts = [
        "A",
        random.choice([p1_parts[1], p2_parts[1]]),
        "for",
        random.choice([p1_parts[3], p2_parts[3]]),
        random.choice([p1_parts[4], p2_parts[4]]),
        random.choice([p1_parts[5], p2_parts[5]])
    ]
    return " ".join(child_parts)

# Mutate an idea
def mutate(idea):
    parts = idea.split()
    if random.random() < 0.1:
        parts[1] = weighted_choice(technologies, tech_seeds)
    if random.random() < 0.1:
        parts[3] = weighted_choice(verbs, verb_seeds)
    if random.random() < 0.1:
        parts[4] = weighted_choice(industries, industry_seeds)
    if random.random() < 0.1:
        parts[5] = weighted_choice(nouns, noun_seeds)
    return " ".join(parts)

# Evaluate idea using LM Studio API (optimized for Gemma)
def evaluate_idea(idea, client, model, max_retries=3):
    system_prompt = """You are an AI evaluating ideas for programmatic implementation. For each idea (e.g., "A Parser for Processing Analytics Data"), return a JSON object with these exact keys and integer ranges:
- "viability": 0-300 (0 = impossible now, 300 = easily feasible)
- "reasoning": a short string explaining the scores (2-3 sentences)
- "value_potential": 0-100 (0 = no benefit, 100 = very valuable)
- "simplicity": 0-50 (0 = very hard to code, 50 = very easy)
- "novelty": 0-25 (0 = common, 25 = very unique)
- "scalability": 0-25 (0 = no growth, 25 = highly scalable)

Steps:
1. Viability: Can it use common tools (e.g., APIs, databases)? 200-300 if yes, 0-100 if not.
2. Value Potential: Does it solve a problem? 80-100 for big impact, 20-40 for small.
3. Simplicity: Easy with basic coding skills? 40-50 if yes, 0-20 if hard.
4. Novelty: Is it new? 20-25 if unique, 0-10 if common.
5. Scalability: Can it grow easily? 20-25 if yes, 0-10 if not.

Return only valid JSON, nothing else. Example:
{"viability": 270, "reasoning": "A Parser for Processing Analytics Data uses existing libraries and APIs, so it’s feasible. It helps with data insights and is easy to code, but it’s not very new.", "value_potential": 80, "simplicity": 45, "novelty": 20, "scalability": 22}"""
    start_time = time.time()
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Evaluate this idea: {idea}"}
                ],
                temperature=0.5,
                max_tokens=400
            )
            content = completion.choices[0].message.content.strip()
            print(f"Raw response for '{idea}': {content}")  # Debug output
            if not content:
                raise ValueError("Empty response from model")
            # Strip ```json markers if present
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            evaluation = json.loads(content)
            end_time = time.time()
            return evaluation, end_time - start_time
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)
            else:
                end_time = time.time()
                return {
                    "viability": 0,
                    "reasoning": f"Failed to evaluate '{idea}' due to invalid response.",
                    "value_potential": 0,
                    "simplicity": 0,
                    "novelty": 0,
                    "scalability": 0
                }, end_time - start_time
        except Exception as e:
            print(f"LM Studio Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)
            else:
                end_time = time.time()
                return None, end_time - start_time

# Calculate fitness score
def calculate_fitness(evaluation):
    if evaluation is None:
        return 0
    score = 0
    score += evaluation.get('viability', 0) if evaluation.get('viability') is not None else 0
    score += evaluation.get('value_potential', 0) if evaluation.get('value_potential') is not None else 0
    score += evaluation.get('simplicity', 0) if evaluation.get('simplicity') is not None else 0
    score += evaluation.get('novelty', 0) if evaluation.get('novelty') is not None else 0
    score += evaluation.get('scalability', 0) if evaluation.get('scalability') is not None else 0
    return score

# Main function with quartile-based genetic algorithm
def main():
    print("Welcome to the Idea Generator Genetic Algorithm!")
    print("Using LM Studio with Gemma.")

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model = "gemma-2-1b"  # Adjust to "gemma-3-1b" if that's your actual model
    evaluate_idea_fn = lambda idea: evaluate_idea(idea, client, model)

    csv_file = "idea_log.csv"
    population_size = 100
    quartile_size = population_size // 4  # 25 per quartile

    print(f"Testing API connectivity with {model}...")
    test_idea = "A Parser for Processing Analytics Data"
    evaluation, eval_time = evaluate_idea_fn(test_idea)
    if evaluation is None:
        print("LM Studio connectivity test failed. Check if server is running at http://localhost:1234.")
        return
    print("API connectivity test successful.")

    csv_headers = ["Timestamp", "Generation", "Gene Sequence", "Fitness Score", "Evaluation Time (s)", "Reasoning"]
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    generation = 0
    population = [(generate_random_idea(), None, 0) for _ in range(population_size)]

    try:
        while True:
            print(f"\nGeneration {generation} (Population: {population_size})")
            
            for i, (idea, _, _) in enumerate(population):
                if population[i][1] is None:
                    evaluation, eval_time = evaluate_idea_fn(idea)
                    fitness_score = calculate_fitness(evaluation)
                    population[i] = (idea, evaluation, fitness_score)
                    
                    timestamp = datetime.datetime.now().isoformat()
                    reasoning = evaluation.get('reasoning', 'No reasoning provided') if evaluation else 'Evaluation failed'
                    csv_row = [timestamp, generation, idea, fitness_score, round(eval_time, 2), reasoning]
                    with open(csv_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(csv_row)
                    
                    print(f"Evaluated {i+1}/{population_size}: {idea}")
                    print(f"Evaluation: {evaluation}")
                    print(f"Fitness Score: {fitness_score}")
                    if fitness_score >= 400:
                        print("*** High Fitness Idea for Coding ***")
                    print(f"Evaluation Time: {eval_time:.2f} seconds")

            population.sort(key=lambda x: x[2], reverse=True)
            
            q1_top = population[:quartile_size]
            print(f"\nTop Quartile (Q1) - Carried Over (Generation {generation}):")
            for idea, _, score in q1_top:
                print(f"- {idea} (Score: {score})")

            q2_offspring = []
            for _ in range(quartile_size):
                parent1, parent2 = random.sample(q1_top, 2)
                child = crossover(parent1[0], parent2[0])
                q2_offspring.append((child, None, 0))

            q3_mutations = []
            for _ in range(quartile_size):
                parent = random.choice(q1_top)
                mutant = mutate(parent[0])
                q3_mutations.append((mutant, None, 0))

            q4_random = [(generate_random_idea(), None, 0) for _ in range(quartile_size)]
            population = q1_top + q2_offspring + q3_mutations + q4_random
            generation += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"\nFinal Top Ideas from Generation {generation-1}:")
        population.sort(key=lambda x: x[2], reverse=True)
        for idea, _, score in population[:quartile_size]:
            print(f"- {idea} (Score: {score})")

if __name__ == "__main__":
    main()