import csv
import time
import random
import datetime
import json
from openai import OpenAI
from tqdm import tqdm
import argparse
import os

# Seed pools
tech_seeds = ["AI", "Blockchain", "Cloud", "IoT", "Quantum", "Robotics", "AR", "VR", "5G", "Edge", "Microservices", "Serverless", "API", "Database", "Neural"]
industry_seeds = ["Health", "Finance", "Education", "Gaming", "Logistics", "Retail", "Energy", "Media", "Travel", "Agritech", "Edtech", "Fintech", "Proptech", "Legal", "Space"]
verb_seeds = ["Analyzes", "Optimizes", "Generates", "Secures", "Tracks", "Predicts", "Automates", "Integrates", "Scales", "Monitors", "Visualizes", "Processes", "Deploys", "Encrypts", "Streams"]
noun_seeds = ["Data", "Platform", "Model", "Network", "System", "Engine", "Framework", "Service", "Application", "Dashboard", "Pipeline", "Cluster", "Interface", "Module", "Sensor"]

def generate_random_idea(available_tech, available_verbs, available_nouns, available_industries, topic=None):
    tech = random.choice(available_tech)
    verb = random.choice(available_verbs)
    noun = random.choice(available_nouns)
    industry = random.choice(available_industries)
    idea = f"A {tech}-powered system that {verb} {noun} in {industry}"
    if topic:
        idea += f" to support {topic}"
    return idea

def crossover(parent1, parent2, available_tech, available_verbs, available_nouns, available_industries, topic=None):
    p1_parts = parent1.split()
    p2_parts = parent2.split()
    tech_idx = 1
    verb_idx = 5
    noun_idx = 6
    in_idx = p1_parts.index("in") if "in" in p1_parts else 8
    industry_idx = in_idx + 1
    
    p1_tech = p1_parts[tech_idx] if len(p1_parts) > tech_idx else random.choice(available_tech)
    p2_tech = p2_parts[tech_idx] if len(p2_parts) > tech_idx else random.choice(available_tech)
    p1_verb = p1_parts[verb_idx] if len(p1_parts) > verb_idx else random.choice(available_verbs)
    p2_verb = p2_parts[verb_idx] if len(p2_parts) > verb_idx else random.choice(available_verbs)
    p1_noun = p1_parts[noun_idx] if len(p1_parts) > noun_idx else random.choice(available_nouns)
    p2_noun = p2_parts[noun_idx] if len(p2_parts) > noun_idx else random.choice(available_nouns)
    p1_industry = p1_parts[industry_idx] if len(p1_parts) > industry_idx else random.choice(available_industries)
    p2_industry = p2_parts[industry_idx] if len(p2_parts) > industry_idx else random.choice(available_industries)

    tech = random.choice([p1_tech, p2_tech, random.choice(available_tech)])
    verb = random.choice([p1_verb, p2_verb, random.choice(available_verbs)])
    noun = random.choice([p1_noun, p2_noun, random.choice(available_nouns)])
    industry = random.choice([p1_industry, p2_industry, random.choice(available_industries)])
    
    idea = f"A {tech}-powered system that {verb} {noun} in {industry}"
    if topic:
        idea += f" to support {topic}"
    
    return idea

def mutate(idea, mutation_rate, available_tech, available_verbs, available_nouns, available_industries, topic=None):
    parts = idea.split()
    if random.random() < mutation_rate:
        parts[1] = random.choice(available_tech)
    if random.random() < mutation_rate:
        parts[5] = random.choice(available_verbs)
    if random.random() < mutation_rate:
        parts[6] = random.choice(available_nouns)
    if random.random() < mutation_rate and "in" in parts:
        parts[parts.index("in") + 1] = random.choice(available_industries)
    idea = " ".join(parts)
    if topic and not idea.endswith(f"to support {topic}"):
        idea += f" to support {topic}"
    return idea

def evaluate_idea(idea, client, model):
    prompt = f"""You are a pragmatic, no-nonsense tech strategist with a razor-sharp eye for detail and zero tolerance for fluff. Given the idea: '{idea}', return a JSON object with these exact keys and integer ranges:
- "viability": 0-100 (score based on technical feasibility and real-world execution; proven concepts can exceed 50, plausible but untested ideas land 30-50)
- "value_potential": 0-100 (score based on market demand and revenue potential; broad, tangible appeal scores high, vague or niche scores low)
- "simplicity": 0-100 (score based on ease of implementation; clean and clear scores above 50, tangled or opaque scores below 30)
- "novelty": 0-100 (score based on innovation; genuinely fresh scores above 50, tired retreads score below 40)
- "scalability": 0-100 (score based on growth potential across markets; wide reach scores high, narrow focus scores low)
- "cringe": 0 or 1 (1 if the idea is grammatically broken, logically incoherent, or reeks of word-vomit—like 'Visualizes Sensor' or 'Encrypts Cluster' that don’t form a proper sentence; 0 if it’s structurally sound, even if quirky or bold)
- "reasoning": a string that flatly explains the idea and why it scored what it did (keep it concise, focus on the concept and score justification, avoid repetition or flair—no two explanations should start the same way)
Return only the JSON object as a string. Be ruthless with grammatical nonsense—flag it—but don’t fuss over minor oddities that still make sense. Stick to the facts."""
    start = time.time()
    try:
        completion = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400)
        content = completion.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        eval = json.loads(content)
        for key in ['viability', 'value_potential', 'simplicity', 'novelty', 'scalability', 'cringe']:
            eval[key] = int(eval[key])
        return eval, time.time() - start, None
    except Exception as e:
        return {"viability": 0, "value_potential": 0, "simplicity": 0, "novelty": 0, "scalability": 0, "cringe": 1, "reasoning": str(e)}, time.time() - start, str(e)

def generate_alternative(idea, client, model):
    prompt = f"""Given the idea: '{idea}', rewrite it into a clear, practical, and coherent single sentence using this exact template: 'A {{tech}}-powered system that {{verb}} {{noun}} in {{industry}}' (plus ' to support {{topic}}' if applicable). Replace {{tech}}, {{verb}}, {{noun}}, and {{industry}} with meaningful words that preserve the original idea’s innovative essence, avoiding awkward or nonsensical phrasing. Return only the new idea as a single short sentence."""
    try:
        completion = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=0.7, max_tokens=100)
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Rewrite failed for '{idea}': {str(e)}")
        return idea

def calculate_fitness(eval, generation, max_generations):
    weights = {"viability": 2.0, "novelty": 2.0 + (generation / max_generations) * 0.5, "value_potential": 1.0, "simplicity": 0.8, "scalability": 0.8}
    score = sum(int(eval[k]) * weights[k] for k in weights)
    if eval.get("cringe", 0) == 1:
        score -= min(50, score * 0.2)
    return max(0, score)

def log_idea(csv_file, generation, idea, eval, fitness, eval_time, cringe_attempts, error=None):
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), generation, idea, fitness,
                         eval.get('viability', 0), eval.get('value_potential', 0), eval.get('simplicity', 0),
                         eval.get('novelty', 0), eval.get('scalability', 0), eval.get('cringe', 0),
                         round(eval_time, 2), eval.get('reasoning', ''), cringe_attempts, error or ''])
        f.flush()

def initialize_csv(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Generation", "Gene Sequence", "Fitness Score", "Viability",
                             "Value Potential", "Simplicity", "Novelty", "Scalability", "Cringe",
                             "Evaluation Time (s)", "Reasoning", "Cringe Attempts", "Error"])

def main():
    parser = argparse.ArgumentParser(description="Idea Generator GA")
    parser.add_argument('--population_size', type=int, default=4, help="Number of ideas per generation")
    parser.add_argument('--max_generations', type=int, default=50, help="Number of generations")
    parser.add_argument('--topic', type=str, default=None, help="Optional topic for all ideas (e.g., 'sustainability')")
    args = parser.parse_args()

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model = "gemma-3-1b"
    csv_file = f"idea_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    initialize_csv(csv_file)
    seen_ideas = set()

    random.seed(datetime.datetime.now().timestamp())

    for generation in range(args.max_generations + 1):
        mutation_rate = 0.1 + (generation / args.max_generations) * 0.2
        print(f"\nGeneration {generation} (Mutation Rate: {mutation_rate:.2f})")

        available_tech = tech_seeds.copy()
        available_verbs = verb_seeds.copy()
        available_nouns = noun_seeds.copy()
        available_industries = industry_seeds.copy()

        if generation == 0:
            population = []
            with tqdm(total=args.population_size, desc=f"Generation {generation}") as pbar:
                while len(population) < args.population_size:
                    idea = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries, args.topic)
                    if idea not in seen_ideas:
                        seen_ideas.add(idea)
                        eval, eval_time, error = evaluate_idea(idea, client, model)
                        total_eval_time = eval_time
                        fitness = calculate_fitness(eval, generation, args.max_generations)
                        cringe_attempts = 0
                        if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                            print(f"Cringe flagged: {idea} -> {eval['reasoning']}")
                            new_idea = generate_alternative(idea, client, model)
                            if new_idea not in seen_ideas:
                                print(f"Rewritten to: {new_idea}")
                                idea = new_idea
                                eval, eval_time, error = evaluate_idea(idea, client, model)
                                total_eval_time += eval_time
                                fitness = calculate_fitness(eval, generation, args.max_generations)
                                seen_ideas.add(idea)
                                cringe_attempts += 1
                        population.append((idea, eval, fitness, cringe_attempts, total_eval_time, error))
                        log_idea(csv_file, generation, idea, eval, fitness, total_eval_time, cringe_attempts, error)
                        pbar.update(1)
        else:
            population = [(idea, eval, calculate_fitness(eval, generation, args.max_generations), 
                           cringe_attempts, total_eval_time, error) 
                          for idea, eval, fitness, cringe_attempts, total_eval_time, error in population]
            population = sorted(population, key=lambda x: x[2], reverse=True)

            top_size = max(1, args.population_size // 4)
            next_population = population[:top_size]
            remaining = args.population_size - len(next_population)
            crossover_size = min(remaining // 3, remaining)
            mutation_size = min(remaining // 3, remaining - crossover_size)
            random_size = remaining - crossover_size - mutation_size
            viable = [p for p in population if p[3] < 2]

            with tqdm(total=remaining, desc=f"Generating Gen {generation}") as pbar:
                for _ in range(crossover_size):
                    if len(next_population) >= args.population_size:
                        break
                    if len(viable) >= 2:
                        child = crossover(random.choice(viable)[0], random.choice(viable)[0],
                                        available_tech, available_verbs, available_nouns, available_industries, args.topic)
                        if child not in seen_ideas:
                            seen_ideas.add(child)
                            eval, eval_time, error = evaluate_idea(child, client, model)
                            total_eval_time = eval_time
                            fitness = calculate_fitness(eval, generation, args.max_generations)
                            cringe_attempts = 0
                            if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                print(f"Cringe flagged: {child} -> {eval['reasoning']}")
                                new_child = generate_alternative(child, client, model)
                                if new_child not in seen_ideas:
                                    print(f"Rewritten to: {new_child}")
                                    child = new_child
                                    eval, eval_time, error = evaluate_idea(child, client, model)
                                    total_eval_time += eval_time
                                    fitness = calculate_fitness(eval, generation, args.max_generations)
                                    seen_ideas.add(child)
                                    cringe_attempts += 1
                            next_population.append((child, eval, fitness, cringe_attempts, total_eval_time, error))
                            log_idea(csv_file, generation, child, eval, fitness, total_eval_time, cringe_attempts, error)
                            pbar.update(1)

                for _ in range(mutation_size):
                    if len(next_population) >= args.population_size:
                        break
                    if viable:
                        mutant = mutate(random.choice(viable)[0], mutation_rate,
                                       available_tech, available_verbs, available_nouns, available_industries, args.topic)
                        if mutant not in seen_ideas:
                            seen_ideas.add(mutant)
                            eval, eval_time, error = evaluate_idea(mutant, client, model)
                            total_eval_time = eval_time
                            fitness = calculate_fitness(eval, generation, args.max_generations)
                            cringe_attempts = 0
                            if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                print(f"Cringe flagged: {mutant} -> {eval['reasoning']}")
                                new_mutant = generate_alternative(mutant, client, model)
                                if new_mutant not in seen_ideas:
                                    print(f"Rewritten to: {new_mutant}")
                                    mutant = new_mutant
                                    eval, eval_time, error = evaluate_idea(mutant, client, model)
                                    total_eval_time += eval_time
                                    fitness = calculate_fitness(eval, generation, args.max_generations)
                                    seen_ideas.add(mutant)
                                    cringe_attempts += 1
                            next_population.append((mutant, eval, fitness, cringe_attempts, total_eval_time, error))
                            log_idea(csv_file, generation, mutant, eval, fitness, total_eval_time, cringe_attempts, error)
                            pbar.update(1)

                for _ in range(random_size):
                    if len(next_population) >= args.population_size:
                        break
                    idea = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries, args.topic)
                    if idea not in seen_ideas:
                        seen_ideas.add(idea)
                        eval, eval_time, error = evaluate_idea(idea, client, model)
                        total_eval_time = eval_time
                        fitness = calculate_fitness(eval, generation, args.max_generations)
                        cringe_attempts = 0
                        if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                            print(f"Cringe flagged: {idea} -> {eval['reasoning']}")
                            new_idea = generate_alternative(idea, client, model)
                            if new_idea not in seen_ideas:
                                print(f"Rewritten to: {new_idea}")
                                idea = new_idea
                                eval, eval_time, error = evaluate_idea(idea, client, model)
                                total_eval_time += eval_time
                                fitness = calculate_fitness(eval, generation, args.max_generations)
                                seen_ideas.add(idea)
                                cringe_attempts += 1
                        next_population.append((idea, eval, fitness, cringe_attempts, total_eval_time, error))
                        log_idea(csv_file, generation, idea, eval, fitness, total_eval_time, cringe_attempts, error)
                        pbar.update(1)

            population = next_population[:args.population_size]

        avg_fitness = sum(p[2] for p in population) / len(population) if population else 0
        best_fitness = max(p[2] for p in population) if population else 0
        print(f"Average Fitness: {avg_fitness:.2f} | Best Fitness: {best_fitness:.2f}")

if __name__ == "__main__":
    main()