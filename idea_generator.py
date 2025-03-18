import csv
import time
import random
import datetime
import os
import json
import shutil
import nltk
from nltk.corpus import words
from openai import OpenAI
from tqdm import tqdm
import argparse

nltk.download('words', quiet=True)
english_words = set(words.words())

# Curated seed pools
tech_seeds = ["API", "Database", "Algorithm", "Interface", "Server", "Cloud Computing", "Network", "Module", "Automation", "Scripting", "Blockchain", "Machine Learning", "Artificial Intelligence", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning", "Quantum Computing", "Edge Computing", "5G", "Internet of Things", "Augmented Reality", "Virtual Reality", "Mixed Reality", "Web3", "Decentralized Finance", "Non-Fungible Tokens", "Metaverse", "Containerization", "Orchestration", "Continuous Integration", "Continuous Delivery", "Infrastructure as Code", "Configuration Management", "Monitoring", "Logging", "Alerting", "Incident Response", "Service Mesh", "Cybersecurity", "Penetration Testing", "Vulnerability Assessment", "Threat Intelligence", "Identity Management", "Encryption", "Intrusion Detection", "Security Analytics", "Big Data", "Data Lake", "Data Warehouse", "Data Pipeline", "Data Streaming", "Real-time Analytics", "Batch Processing", "Data Visualization", "Business Intelligence", "Microservices", "Serverless", "Event-Driven Architecture", "Service-Oriented Architecture", "Monolithic Architecture", "RESTful Services", "GraphQL Services", "Real-time Communication", "Publish-Subscribe Systems"]
industry_seeds = ["Productivity", "Health", "Education", "Finance", "Gaming", "Travel", "Social", "Ecommerce", "Media", "Security", "Analytics", "Logistics", "Entertainment", "Communication", "Retail", "HR", "Marketing", "Energy", "Research", "Telemedicine", "Wearable Health Tech", "Genomics", "Mental Health", "Digital Therapeutics", "Fintech", "Cryptocurrency", "Insurtech", "Regtech", "Wealthtech", "Payment Processing", "Edtech", "Online Learning", "Educational Games", "Language Learning", "Skill Development", "Agritech", "Precision Agriculture", "Smart Farming", "Agricultural Drones", "Vertical Farming", "Clean Energy", "Renewable Energy", "Energy Storage", "Smart Grids", "Carbon Capture", "Space Tech", "Satellite Communications", "Space Tourism", "Asteroid Mining", "Space Debris Management", "Transportation", "Autonomous Vehicles", "Electric Vehicles", "Ride Sharing", "Mobility as a Service", "Construction Tech", "Building Information Modeling", "3D Printing in Construction", "Smart Buildings", "Real Estate Tech", "Proptech", "Virtual Tours", "Property Management Software", "Food Tech", "Alternative Proteins", "Food Delivery", "Restaurant Management", "Nutrition Tracking", "Fashion Tech", "Virtual Fitting Rooms", "Sustainable Fashion", "Fashion E-commerce", "Sports Tech", "Wearable Fitness Tech", "Esports", "Sports Analytics", "Fan Engagement", "Legal Tech", "Contract Management", "Legal Research", "E-Discovery", "Compliance", "Govtech", "Digital Government Services", "Civic Tech", "Smart Cities", "Public Safety"]
verb_seeds = ["Tracks", "Analyzes", "Processes", "Automates", "Deploys", "Optimizes", "Secures", "Predicts", "Monitors", "Integrates", "Scales", "Encrypts", "Visualizes", "Orchestrates", "Classifies", "Filters", "Displays", "Notifies", "Converts", "Organizes", "Shares", "Syncs", "Generates", "Renders", "Streams", "Fetches", "Updates", "Parses", "Logs", "Schedules", "Pushes", "Pulls", "Extracts", "Transforms", "Loads", "Queries", "Indexes", "Searches", "Aggregates", "Correlates", "Anonymizes", "Tokenizes", "Hashes", "Signs", "Verifies", "Backs up", "Restores", "Archives", "Compresses", "Decompresses", "Encodes", "Decodes", "Transcodes", "Buffers", "Throttles", "Routes", "Forwards", "Proxies", "Reverse Proxies", "Load Balances", "Failovers", "Replicates", "Shards", "Partitions", "Migrates", "Upgrades", "Patches", "Configures", "Provisions", "Deprovisions", "Autoscales", "Rightsizes", "Tunes", "Benchmarks", "Profiles", "Traces", "Instruments", "Alerts", "Escalates", "Remediates", "Forecasts", "Projects", "Extrapolates", "Interpolates", "Approximates", "Estimates", "Calculates", "Computes", "Solves", "Resolves", "Minimizes", "Maximizes", "Equalizes", "Normalizes", "Standardizes", "Composes", "Decomposes", "Abstracts", "Encapsulates", "Modularizes", "Refactors", "Rewrites", "Ports", "Emulates"]
noun_seeds = ["System", "Platform", "Framework", "Library", "Application", "Portal", "Simulator", "Emulator", "Compiler", "Interpreter", "IDE", "API Gateway", "Message Queue", "Cache Layer", "Load Balancer", "Firewall", "Proxy", "Reverse Proxy", "Container", "Cluster", "Pod", "Microservice", "Monolith", "Data", "Database Schema", "Data Lake", "Data Warehouse", "ETL Pipeline", "CI/CD Pipeline", "Version Control", "Git Repository", "Neural Network", "Deep Learning Model", "Natural Language Processor", "Computer Vision System", "Reinforcement Learning Agent", "Subscription Service", "Freemium Model", "Ad-Supported Platform", "Marketplace", "Platform as a Service", "Software as a Service", "Infrastructure as a Service", "API Economy", "Open Source Project", "Crowdsourcing Platform", "Gamification System", "Personalization Engine", "Recommendation System", "Chatbot", "Virtual Assistant", "Dashboard", "Report", "Feed", "Alert", "Profile", "List", "Stream", "Widget", "Engine", "Service", "Model", "Plugin", "Queue", "Graph", "Log", "Form", "Map", "Index", "Cache", "Flow", "Node", "Edge Computing Device", "5G Network", "Quantum Computer", "Blockchain Network", "Decentralized Finance Platform", "Non-Fungible Token Marketplace", "Metaverse Environment", "Digital Twin", "Autonomous Vehicle", "Smart City Infrastructure", "Precision Agriculture System", "Renewable Energy Grid", "Penetration Testing Tool", "Vulnerability Scanner", "Threat Intelligence Platform", "Identity and Access Management System", "Encryption Library", "Firewall Appliance", "Intrusion Detection System", "SIEM Solution", "Continuous Integration Server", "Continuous Delivery Pipeline", "Infrastructure as Code Tool", "Configuration Management System", "Monitoring Dashboard", "Logging Service", "Alerting System", "Incident Response Platform", "Service Mesh", "Container Orchestrator"]

# Expand pools with NLTK words for variety
def expand_pool(seed_list, target_size=1000):
    pool = set(seed_list)
    candidates = [w for w in english_words if len(w) > 3 and w[0].isupper() == False]  # Avoid proper nouns
    while len(pool) < target_size and candidates:
        word = random.choice(candidates)
        pool.add(word)
        candidates.remove(word)
    if len(pool) < target_size:
        pool.update(random.choices(seed_list, k=target_size - len(pool)))
    return list(pool)

technologies = expand_pool(tech_seeds)
industries = expand_pool(industry_seeds)
verbs = expand_pool(verb_seeds)
nouns = expand_pool(noun_seeds)

def weighted_choice(pool, seeds, seed_weight=0.6):
    return random.choice(seeds) if random.random() < seed_weight else random.choice(pool)

def generate_random_idea():
    tech = weighted_choice(technologies, tech_seeds)
    verb = weighted_choice(verbs, verb_seeds)
    noun = weighted_choice(nouns, noun_seeds)
    industry = weighted_choice(industries, industry_seeds)
    return f"A {tech}-powered system that {verb} {noun} in {industry}"

def crossover(parent1, parent2):
    p1_parts = parent1.split()
    p2_parts = parent2.split()
    child_parts = [
        "A",
        random.choice([p1_parts[1], p2_parts[1]]),
        "powered",
        "system",
        "that",
        random.choice([p1_parts[5], p2_parts[5]]),
        random.choice([p1_parts[6], p2_parts[6]]),
        "in",
        random.choice([p1_parts[8], p2_parts[8]])
    ]
    return " ".join(child_parts)

def mutate(idea, mutation_rate):
    parts = idea.split()
    if random.random() < mutation_rate:
        parts[1] = weighted_choice(technologies, tech_seeds)
    if random.random() < mutation_rate:
        parts[5] = weighted_choice(verbs, verb_seeds)
    if random.random() < mutation_rate:
        parts[6] = weighted_choice(nouns, noun_seeds)
    if random.random() < mutation_rate:
        parts[8] = weighted_choice(industries, industry_seeds)
    return " ".join(parts)

def evaluate_idea(idea, client, model, max_retries=3):
    system_prompt = f"""You are an AI evaluating ideas for programmatic implementation with strict, precise standards. For each idea, return a JSON object with these exact keys and integer ranges (all 0-100):
- "viability": 0-100 (90-100 for proven tech with clear implementation paths; 70-89 for feasible but challenging; 50-69 for speculative with potential; 0-49 if impractical or unproven)
- "reasoning": a detailed string explaining the scores (3-4 sentences, addressing each metric)
- "value_potential": 0-100 (90-100 for transformative impact on a major problem; 70-89 for significant practical use; 50-69 for moderate benefit; 0-49 if niche or low impact)
- "simplicity": 0-100 (90-100 for minimal coding effort with existing tools; 70-89 for manageable complexity; 50-69 for moderate difficulty; 0-49 if highly complex or resource-intensive)
- "novelty": 0-100 (90-100 for truly groundbreaking concepts; 70-89 for unique twists on existing ideas; 50-69 for somewhat original; 0-49 if common or derivative)
- "scalability": 0-100 (90-100 for massive, seamless growth potential; 70-89 for good scalability with effort; 50-69 for limited but viable scaling; 0-49 if constrained or unscalable)
- "cringe": 1 or 0 (1 = awkward phrasing, unprofessional tone, or nonsensical/incoherent combinations like 'Cryptomorphic Models in Pathologicoclinical'; 0 = clear, professional, and coherent)

Steps:
1. Viability: Assess if the technology is proven and fits the task/industry. Flag obscure terms (e.g., 'Haulier', 'Cryptomorphic') as less viable unless contextually clear.
2. Value Potential: Evaluate the problem’s size and solution’s impact.
3. Simplicity: Judge coding effort based on current tech.
4. Novelty: Check uniqueness—give credit to weird terms if they suggest an innovative angle.
5. Scalability: Estimate growth capacity.
6. Cringe: Set to 1 if the idea is incoherent, uses mismatched or obscure terms without clear meaning (e.g., 'word vomit'), or sounds unprofessional; 0 if it makes sense.

Return only valid JSON."""
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
            if not content:
                raise ValueError("Empty response from model")
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            evaluation = json.loads(content.strip())
            end_time = time.time()
            return evaluation, end_time - start_time
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing response for '{idea}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                end_time = time.time()
                return {"viability": 0, "reasoning": f"Failed to evaluate: {str(e)}", "value_potential": 0, "simplicity": 0, "novelty": 0, "scalability": 0, "cringe": 1}, end_time - start_time
        except Exception as e:
            print(f"API Error for '{idea}': {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                end_time = time.time()
                return {"viability": 0, "reasoning": f"API failure: {str(e)}", "value_potential": 0, "simplicity": 0, "novelty": 0, "scalability": 0, "cringe": 1}, end_time - start_time

def generate_alternative(idea, client, model):
    system_prompt = f"""Given an idea that may contain obscure or nonsensical terms (e.g., 'word vomit' like 'Cryptomorphic Models' or 'Pathologicoclinical'), rewrite it into a practical, general, and coherent version that preserves the innovative essence. The new idea should:
- Interpret the weird terms into a meaningful tech/action/object/industry concept (e.g., 'Cryptomorphic' might suggest complex pattern analysis, 'Haulier' might imply logistics tech).
- Use natural, professional phrasing without sticking to a strict template, but stay close to a structure involving a technology, an action, an object, and an industry/context.
- Avoid generic overcorrections—keep the creative spark.
Return only the new idea as a plain string.

Examples:
Input: "A Service Mesh-powered system that Routes Cryptomorphic Models in Pathologicoclinical"
Output: "A Service Mesh solution routing advanced data patterns in healthcare diagnostics"

Input: "A Haulier-powered system that Extracts Information from Agricultural Texts in Achira"
Output: "A logistics-driven platform extracting insights from agricultural data"

Input: "A Blockchain-powered system that Fetches Goat Insights in Agritech"
Output: "A Blockchain system retrieving livestock data for agricultural innovation"
"""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Suggest an alternative for: {idea}"}
            ],
            temperature=0.6,
            max_tokens=50
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating alternative for '{idea}': {e}")
        return idea

def meets_rewrite_thresholds(evaluation):
    return evaluation.get('viability', 0) >= 70 and evaluation.get('novelty', 0) >= 60

def calculate_fitness(evaluation, generation, max_generations):
    if not evaluation:
        return 0
    viability = evaluation.get('viability', 0)
    value_potential = evaluation.get('value_potential', 0)
    simplicity = evaluation.get('simplicity', 0)
    novelty = evaluation.get('novelty', 0)
    scalability = evaluation.get('scalability', 0)
    cringe = evaluation.get('cringe', 0)
    viability_weight = 2.0
    novelty_weight = 2.0 + (generation / max_generations) * 0.5
    value_weight = 1.0
    simplicity_weight = 0.8
    scalability_weight = 0.8
    score = (viability * viability_weight + value_potential * value_weight + 
             simplicity * simplicity_weight + novelty * novelty_weight + scalability * scalability_weight)
    if cringe == 1:
        score -= min(50, score * 0.2)
    return max(0, score)

def archive_csv(csv_file):
    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived_file = os.path.join(archive_dir, f"idea_log_{timestamp}.csv")
    if os.path.exists(csv_file):
        shutil.copy(csv_file, archived_file)
        print(f"Archived CSV as: {archived_file}")
    return archived_file

def tournament_selection(population, tournament_size=5):
    participants = random.sample(population, min(tournament_size, len(population)))
    return max(participants, key=lambda x: x[2])

def generate_and_evaluate(idea, client, model, seen_ideas):
    attempts = 0
    while idea in seen_ideas and attempts < 20:
        idea = generate_random_idea()
        attempts += 1
    eval, eval_time = evaluate_idea(idea, client, model)
    seen_ideas.add(idea)
    return idea, eval, eval_time, attempts

def main():
    parser = argparse.ArgumentParser(description="Idea Generator Genetic Algorithm")
    parser.add_argument('--population_size', type=int, default=100, help="Size of the population")
    parser.add_argument('--max_generations', type=int, default=50, help="Maximum number of generations")
    args = parser.parse_args()

    population_size = args.population_size
    max_generations = args.max_generations
    quartile_size = max(1, population_size // 4)
    initial_mutation_rate = 0.1

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model = "gemma-3-1b"

    csv_file = "idea_log.csv"
    csv_headers = ["Timestamp", "Generation", "Gene Sequence", "Fitness Score", "Viability", 
                   "Value Potential", "Simplicity", "Novelty", "Scalability", "Cringe", 
                   "Evaluation Time (s)", "Reasoning", "Cringe Attempts"]
    
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    test_idea = "A Machine Learning-powered system that Analyzes Data in Healthcare"
    evaluation, _ = evaluate_idea(test_idea, client, model)
    if not evaluation:
        print("API connectivity test failed.")
        return
    print("API connectivity test successful.")

    population = []
    seen_ideas = set()
    for _ in tqdm(range(population_size), desc="Initial Population (Generation 0)"):
        idea = generate_random_idea()
        idea, eval, eval_time, attempts = generate_and_evaluate(idea, client, model, seen_ideas)
        fitness = calculate_fitness(eval, 0, max_generations)
        cringe_attempts = attempts
        if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
            idea = generate_alternative(idea, client, model)
            eval, eval_time = evaluate_idea(idea, client, model)
            seen_ideas.add(idea)
            fitness = calculate_fitness(eval, 0, max_generations)
            cringe_attempts += 1
            if eval.get("cringe", 0) == 1 and meets_rewrite_thresholds(eval):
                idea = generate_alternative(idea, client, model)
                eval, eval_time = evaluate_idea(idea, client, model)
                seen_ideas.add(idea)
                fitness = calculate_fitness(eval, 0, max_generations)
                cringe_attempts += 1
        population.append((idea, eval, fitness, cringe_attempts))
        
        timestamp = datetime.datetime.now().isoformat()
        viability = eval.get('viability', 0)
        value_potential = eval.get('value_potential', 0)
        simplicity = eval.get('simplicity', 0)
        novelty = eval.get('novelty', 0)
        scalability = eval.get('scalability', 0)
        cringe = eval.get('cringe', 0)
        reasoning = eval.get('reasoning', 'No reasoning provided')
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, 0, idea, fitness, viability, value_potential, simplicity, 
                             novelty, scalability, cringe, round(eval_time, 2), reasoning, cringe_attempts])

    generation = 1
    avg_viability_history = []
    avg_novelty_history = []

    while True:
        mutation_rate = initial_mutation_rate + ((generation - 1) / max_generations) * 0.2
        print(f"\nGeneration {generation} (Mutation Rate: {mutation_rate:.2f})")
        
        for i in tqdm(range(len(population)), desc=f"Evaluating Gen {generation}"):
            idea, eval, fitness, cringe_attempts = population[i]
            eval, eval_time = evaluate_idea(idea, client, model)
            fitness = calculate_fitness(eval, generation, max_generations)
            if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                idea = generate_alternative(idea, client, model)
                eval, eval_time = evaluate_idea(idea, client, model)
                seen_ideas.add(idea)
                fitness = calculate_fitness(eval, generation, max_generations)
                cringe_attempts += 1
                if eval.get("cringe", 0) == 1 and meets_rewrite_thresholds(eval):
                    idea = generate_alternative(idea, client, model)
                    eval, eval_time = evaluate_idea(idea, client, model)
                    seen_ideas.add(idea)
                    fitness = calculate_fitness(eval, generation, max_generations)
                    cringe_attempts += 1
            population[i] = (idea, eval, fitness, cringe_attempts)
            
            timestamp = datetime.datetime.now().isoformat()
            viability = eval.get('viability', 0)
            value_potential = eval.get('value_potential', 0)
            simplicity = eval.get('simplicity', 0)
            novelty = eval.get('novelty', 0)
            scalability = eval.get('scalability', 0)
            cringe = eval.get('cringe', 0)
            reasoning = eval.get('reasoning', 'No reasoning provided')
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, generation, idea, fitness, viability, value_potential, 
                                 simplicity, novelty, scalability, cringe, round(eval_time, 2), 
                                 reasoning, cringe_attempts])

        population.sort(key=lambda x: x[2], reverse=True)
        viable_population = [p for p in population if p[3] < 2]
        q1_size = min(quartile_size, len(viable_population)) if viable_population else 0
        q1_top = viable_population[:q1_size]
        
        remaining_size = population_size - q1_size
        q2_size = remaining_size // 3
        q3_size = remaining_size // 3
        q4_size = remaining_size - q2_size - q3_size
        
        q2_offspring = []
        for _ in range(q2_size):
            if len(q1_top) >= 2:
                parent1 = tournament_selection(q1_top)
                parent2 = tournament_selection(q1_top)
                child = crossover(parent1[0], parent2[0])
                eval, eval_time = evaluate_idea(child, client, model)
                fitness = calculate_fitness(eval, generation, max_generations)
                if eval.get("cringe", 0) == 1:
                    child = generate_alternative(child, client, model)
                    eval, eval_time = evaluate_idea(child, client, model)
                    fitness = calculate_fitness(eval, generation, max_generations)
                seen_ideas.add(child)
                q2_offspring.append((child, eval, fitness, 0))
            else:
                idea = generate_random_idea()
                idea, eval, eval_time, _ = generate_and_evaluate(idea, client, model, seen_ideas)
                fitness = calculate_fitness(eval, generation, max_generations)
                if eval.get("cringe", 0) == 1:
                    idea = generate_alternative(idea, client, model)
                    eval, eval_time = evaluate_idea(idea, client, model)
                    fitness = calculate_fitness(eval, generation, max_generations)
                seen_ideas.add(idea)
                q2_offspring.append((idea, eval, fitness, 0))
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().isoformat(), generation, idea, fitness, 
                                 eval.get('viability', 0), eval.get('value_potential', 0), 
                                 eval.get('simplicity', 0), eval.get('novelty', 0), 
                                 eval.get('scalability', 0), eval.get('cringe', 0), 
                                 round(eval_time, 2), eval.get('reasoning', ''), 0])

        q3_mutations = []
        for _ in range(q3_size):
            if q1_top:
                parent = tournament_selection(q1_top)
                mutant = mutate(parent[0], mutation_rate)
                eval, eval_time = evaluate_idea(mutant, client, model)
                fitness = calculate_fitness(eval, generation, max_generations)
                if eval.get("cringe", 0) == 1:
                    mutant = generate_alternative(mutant, client, model)
                    eval, eval_time = evaluate_idea(mutant, client, model)
                    fitness = calculate_fitness(eval, generation, max_generations)
                seen_ideas.add(mutant)
                q3_mutations.append((mutant, eval, fitness, 0))
            else:
                idea = generate_random_idea()
                idea, eval, eval_time, _ = generate_and_evaluate(idea, client, model, seen_ideas)
                fitness = calculate_fitness(eval, generation, max_generations)
                if eval.get("cringe", 0) == 1:
                    idea = generate_alternative(idea, client, model)
                    eval, eval_time = evaluate_idea(idea, client, model)
                    fitness = calculate_fitness(eval, generation, max_generations)
                seen_ideas.add(idea)
                q3_mutations.append((idea, eval, fitness, 0))
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().isoformat(), generation, idea, fitness, 
                                 eval.get('viability', 0), eval.get('value_potential', 0), 
                                 eval.get('simplicity', 0), eval.get('novelty', 0), 
                                 eval.get('scalability', 0), eval.get('cringe', 0), 
                                 round(eval_time, 2), eval.get('reasoning', ''), 0])

        q4_random = []
        for _ in range(q4_size):
            idea = generate_random_idea()
            idea, eval, eval_time, _ = generate_and_evaluate(idea, client, model, seen_ideas)
            fitness = calculate_fitness(eval, generation, max_generations)
            if eval.get("cringe", 0) == 1:
                idea = generate_alternative(idea, client, model)
                eval, eval_time = evaluate_idea(idea, client, model)
                fitness = calculate_fitness(eval, generation, max_generations)
            seen_ideas.add(idea)
            q4_random.append((idea, eval, fitness, 0))
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.datetime.now().isoformat(), generation, idea, fitness, 
                                 eval.get('viability', 0), eval.get('value_potential', 0), 
                                 eval.get('simplicity', 0), eval.get('novelty', 0), 
                                 eval.get('scalability', 0), eval.get('cringe', 0), 
                                 round(eval_time, 2), eval.get('reasoning', ''), 0])

        population = q1_top + q2_offspring + q3_mutations + q4_random
        assert len(population) == population_size, f"Population size mismatch: expected {population_size}, got {len(population)}"

        avg_viability = sum(p[1].get('viability', 0) for p in population) / len(population)
        avg_novelty = sum(p[1].get('novelty', 0) for p in population) / len(population)
        avg_viability_history.append(avg_viability)
        avg_novelty_history.append(avg_novelty)
        if (len(avg_viability_history) > 3 and 
            all(abs(avg_viability_history[-1] - x) < 5 for x in avg_viability_history[-3:-1]) and
            all(abs(avg_novelty_history[-1] - x) < 5 for x in avg_novelty_history[-3:-1])):
            print(f"Early stopping: Viability ({avg_viability:.2f}) and Novelty ({avg_novelty:.2f}) stabilized")
            break

        avg_fitness = sum(p[2] for p in population) / len(population)
        print(f"Average Fitness: {avg_fitness:.2f}, Avg Viability: {avg_viability:.2f}, Avg Novelty: {avg_novelty:.2f}")
        for idea, _, score, _ in q1_top:
            print(f"- {idea} (Score: {score})")
        
        generation += 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user.")
        archive_csv("idea_log.csv")
        population.sort(key=lambda x: x[2], reverse=True)
        print(f"\nFinal Top Ideas from Last Generation:")
        for idea, _, score, _ in population[:quartile_size]:
            print(f"- {idea} (Score: {score})")