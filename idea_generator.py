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

# Download NLTK words if necessary
nltk.download('words', quiet=True)
english_words = set(words.words())

# Expanded tech term pools (no programming languages)
tech_seeds = [
    # Core Technologies
    "API", "Database", "Algorithm", "Interface", "Server", "Cloud Computing", "Network", "Module", "Automation", "Scripting",
    "Blockchain", "Machine Learning", "Artificial Intelligence", "Deep Learning", "Natural Language Processing", "Computer Vision",
    "Reinforcement Learning", "Quantum Computing", "Edge Computing", "5G", "Internet of Things", "Augmented Reality",
    "Virtual Reality", "Mixed Reality", "Web3", "Decentralized Finance", "Non-Fungible Tokens", "Metaverse",
    # DevOps & Infrastructure
    "Containerization", "Orchestration", "Continuous Integration", "Continuous Delivery", "Infrastructure as Code",
    "Configuration Management", "Monitoring", "Logging", "Alerting", "Incident Response", "Service Mesh",
    # Security
    "Cybersecurity", "Penetration Testing", "Vulnerability Assessment", "Threat Intelligence", "Identity Management",
    "Encryption", "Intrusion Detection", "Security Analytics",
    # Data Tech
    "Big Data", "Data Lake", "Data Warehouse", "Data Pipeline", "Data Streaming", "Real-time Analytics", "Batch Processing",
    "Data Visualization", "Business Intelligence",
    # Architectures
    "Microservices", "Serverless", "Event-Driven Architecture", "Service-Oriented Architecture", "Monolithic Architecture",
    "RESTful Services", "GraphQL Services", "Real-time Communication", "Publish-Subscribe Systems"
]

industry_seeds = [
    "Productivity", "Health", "Education", "Finance", "Gaming", "Travel", "Social", "Ecommerce", "Media", "Security", "Analytics",
    "Logistics", "Entertainment", "Communication", "Retail", "HR", "Marketing", "Energy", "Research", "Telemedicine",
    "Wearable Health Tech", "Genomics", "Mental Health", "Digital Therapeutics", "Fintech", "Cryptocurrency", "Insurtech",
    "Regtech", "Wealthtech", "Payment Processing", "Edtech", "Online Learning", "Educational Games", "Language Learning",
    "Skill Development", "Agritech", "Precision Agriculture", "Smart Farming", "Agricultural Drones", "Vertical Farming",
    "Clean Energy", "Renewable Energy", "Energy Storage", "Smart Grids", "Carbon Capture", "Space Tech", "Satellite Communications",
    "Space Tourism", "Asteroid Mining", "Space Debris Management", "Transportation", "Autonomous Vehicles", "Electric Vehicles",
    "Ride Sharing", "Mobility as a Service", "Construction Tech", "Building Information Modeling", "3D Printing in Construction",
    "Smart Buildings", "Real Estate Tech", "Proptech", "Virtual Tours", "Property Management Software", "Food Tech",
    "Alternative Proteins", "Food Delivery", "Restaurant Management", "Nutrition Tracking", "Fashion Tech", "Virtual Fitting Rooms",
    "Sustainable Fashion", "Fashion E-commerce", "Sports Tech", "Wearable Fitness Tech", "Esports", "Sports Analytics",
    "Fan Engagement", "Legal Tech", "Contract Management", "Legal Research", "E-Discovery", "Compliance", "Govtech",
    "Digital Government Services", "Civic Tech", "Smart Cities", "Public Safety"
]

verb_seeds = [
    "Tracks", "Analyzes", "Processes", "Automates", "Deploys", "Optimizes", "Secures", "Predicts", "Monitors", "Integrates",
    "Scales", "Encrypts", "Visualizes", "Orchestrates", "Classifies", "Filters", "Displays", "Notifies", "Converts", "Organizes",
    "Shares", "Syncs", "Generates", "Renders", "Streams", "Fetches", "Updates", "Parses", "Logs", "Schedules", "Pushes", "Pulls",
    "Extracts", "Transforms", "Loads", "Queries", "Indexes", "Searches", "Aggregates", "Correlates", "Anonymizes", "Tokenizes",
    "Hashes", "Signs", "Verifies", "Backs up", "Restores", "Archives", "Compresses", "Decompresses", "Encodes", "Decodes",
    "Transcodes", "Buffers", "Throttles", "Routes", "Forwards", "Proxies", "Reverse Proxies", "Load Balances", "Failovers",
    "Replicates", "Shards", "Partitions", "Migrates", "Upgrades", "Patches", "Configures", "Provisions", "Deprovisions",
    "Autoscales", "Rightsizes", "Tunes", "Benchmarks", "Profiles", "Traces", "Instruments", "Alerts", "Escalates", "Remediates",
    "Forecasts", "Projects", "Extrapolates", "Interpolates", "Approximates", "Estimates", "Calculates", "Computes", "Solves",
    "Resolves", "Minimizes", "Maximizes", "Equalizes", "Normalizes", "Standardizes", "Composes", "Decomposes", "Abstracts",
    "Encapsulates", "Modularizes", "Refactors", "Rewrites", "Ports", "Emulates"
]

noun_seeds = [
    "System", "Platform", "Framework", "Library", "Application", "Portal", "Simulator", "Emulator", "Compiler", "Interpreter",
    "IDE", "API Gateway", "Message Queue", "Cache Layer", "Load Balancer", "Firewall", "Proxy", "Reverse Proxy", "Container",
    "Cluster", "Pod", "Microservice", "Monolith", "Data", "Database Schema", "Data Lake", "Data Warehouse", "ETL Pipeline",
    "CI/CD Pipeline", "Version Control", "Git Repository", "Neural Network", "Deep Learning Model", "Natural Language Processor",
    "Computer Vision System", "Reinforcement Learning Agent", "Subscription Service", "Freemium Model", "Ad-Supported Platform",
    "Marketplace", "Platform as a Service", "Software as a Service", "Infrastructure as a Service", "API Economy",
    "Open Source Project", "Crowdsourcing Platform", "Gamification System", "Personalization Engine", "Recommendation System",
    "Chatbot", "Virtual Assistant", "Dashboard", "Report", "Feed", "Alert", "Profile", "List", "Stream", "Widget", "Engine",
    "Service", "Model", "Plugin", "Queue", "Graph", "Log", "Form", "Map", "Index", "Cache", "Flow", "Node",
    "Edge Computing Device", "5G Network", "Quantum Computer", "Blockchain Network", "Decentralized Finance Platform",
    "Non-Fungible Token Marketplace", "Metaverse Environment", "Digital Twin", "Autonomous Vehicle", "Smart City Infrastructure",
    "Precision Agriculture System", "Renewable Energy Grid", "Penetration Testing Tool", "Vulnerability Scanner",
    "Threat Intelligence Platform", "Identity and Access Management System", "Encryption Library", "Firewall Appliance",
    "Intrusion Detection System", "SIEM Solution", "Continuous Integration Server", "Continuous Delivery Pipeline",
    "Infrastructure as Code Tool", "Configuration Management System", "Monitoring Dashboard", "Logging Service", "Alerting System",
    "Incident Response Platform", "Service Mesh", "Container Orchestrator"
]

# Expand pools to 1000 terms
def expand_pool(seed_list, target_size=1000):
    pool = set(seed_list)
    candidates = [w for w in english_words if len(w) > 3]
    while len(pool) < target_size and candidates:
        word = random.choice(candidates)
        pool.add(word)
        candidates.remove(word)
    return list(pool)

technologies = expand_pool(tech_seeds)
industries = expand_pool(industry_seeds)
verbs = expand_pool(verb_seeds)
nouns = expand_pool(noun_seeds)

# Weighted choice (60% seed weight)
def weighted_choice(pool, seeds, seed_weight=0.6):
    return random.choice(seeds) if random.random() < seed_weight else random.choice(pool)

# Generate a random idea
def generate_random_idea():
    tech = weighted_choice(technologies, tech_seeds)
    verb = weighted_choice(verbs, verb_seeds)
    noun = weighted_choice(nouns, noun_seeds)
    industry = weighted_choice(industries, industry_seeds)
    return f"A {tech}-powered system that {verb} {noun} in {industry}"

# Crossover two parent ideas
def crossover(parent1, parent2):
    p1_parts = parent1.split()
    p2_parts = parent2.split()
    child_parts = [
        "A",
        random.choice([p1_parts[1], p2_parts[1]]),  # tech
        "powered",
        "system",
        "that",
        random.choice([p1_parts[5], p2_parts[5]]),  # verb
        random.choice([p1_parts[6], p2_parts[6]]),  # noun
        "in",
        random.choice([p1_parts[8], p2_parts[8]])   # industry
    ]
    return " ".join(child_parts)

# Mutate an idea
def mutate(idea):
    parts = idea.split()
    if random.random() < 0.1:
        parts[1] = weighted_choice(technologies, tech_seeds)  # tech
    if random.random() < 0.1:
        parts[5] = weighted_choice(verbs, verb_seeds)        # verb
    if random.random() < 0.1:
        parts[6] = weighted_choice(nouns, noun_seeds)        # noun
    if random.random() < 0.1:
        parts[8] = weighted_choice(industries, industry_seeds)  # industry
    return " ".join(parts)

# Evaluate idea
def evaluate_idea(idea, client, model, max_retries=3):
    system_prompt = f"""You are an AI evaluating ideas for programmatic implementation with strict standards. For each idea (e.g., "A Machine Learning-powered system that Analyzes Data in Healthcare"), return a JSON object with these exact keys and integer ranges (all 0-100):
- "viability": 0-100 (80-100 only for ideas with proven, widely-used technologies; 0-50 if speculative)
- "reasoning": a short string explaining the scores (2-3 sentences)
- "value_potential": 0-100 (80-100 only for significant, real-world impact; 0-50 if minor)
- "simplicity": 0-100 (80-100 only for minimal coding effort; 0-50 if complex)
- "novelty": 0-100 (80-100 only for groundbreaking ideas; 0-50 if common)
- "scalability": 0-100 (80-100 only for massive growth potential; 0-50 if limited)
- "cringe": 1 or 0 (1 = sounds awkward, unprofessional, or nonsensical; 0 = sensible)

Steps:
1. Viability: Does it use proven technologies (e.g., Machine Learning, Cloud Computing)? 80-100 if yes and well-established, 0-50 if experimental or impractical.
2. Value Potential: Does it solve a major problem? 80-100 for high impact, 0-50 for niche or trivial.
3. Simplicity: Can it be built with straightforward methods? 80-100 if easy, 0-50 if intricate.
4. Novelty: Is it truly unique? 80-100 for original concepts, 0-50 for iterations.
5. Scalability: Can it scale massively? 80-100 if highly scalable, 0-50 if constrained.
6. Cringe: Is it awkward or unprofessional? 1 if yes, 0 if no.

Return only valid JSON. Example:
{{"viability": 85, "reasoning": "A Machine Learning-powered system that Analyzes Data in Healthcare uses proven ML techniques, making it highly feasible. It offers significant value in medical diagnostics but isn’t groundbreaking.", "value_potential": 80, "simplicity": 75, "novelty": 40, "scalability": 70, "cringe": 0}}
"""
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
            print(f"Raw response for '{idea}': {content}")
            if not content:
                raise ValueError("Empty response from model")
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
                    "scalability": 0,
                    "cringe": 1
                }, end_time - start_time
        except Exception as e:
            print(f"LM Studio Error: {e}")
            if attempt < max_retries - 1:
                print("Retrying...")
                time.sleep(2 ** attempt)
            else:
                end_time = time.time()
                return None, end_time - start_time

# Generate a sensible alternative with novelty focus on second rewrite
def generate_alternative(idea, client, model, preserve_novelty=False):
    if preserve_novelty:
        system_prompt = """Given an awkward or unprofessional idea, suggest a sensible, professional alternative that preserves its core concept and unique, innovative elements. Avoid over-normalizing it into a common idea. Return the new idea as a plain string. Example:
Input: "A Quantum Computing-powered system that Dances Finance Logs in Fintech"
Output: "A Quantum Computing-powered system that Dynamically Analyzes Finance Data in Fintech" """
    else:
        system_prompt = """Given a cringe or awkward-sounding idea, suggest a sensible, professional alternative that preserves the core concept but improves clarity and appeal. Return the new idea as a plain string. Example:
Input: "A Blockchain-powered system that Fetches Goat Insights in Agritech"
Output: "A Blockchain-powered system that Retrieves Livestock Data in Agritech" """
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Suggest a sensible alternative for: {idea}"}
            ],
            temperature=0.7,
            max_tokens=50
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating alternative: {e}")
        return idea

# Check if idea meets thresholds for second rewrite
def meets_rewrite_thresholds(evaluation):
    return (evaluation.get('viability', 0) >= 70 and
            evaluation.get('novelty', 0) >= 50)

# Calculate fitness score
def calculate_fitness(evaluation):
    if evaluation is None:
        return 0
    score = 0
    viability = evaluation.get('viability', 0)
    value_potential = evaluation.get('value_potential', 0)
    simplicity = evaluation.get('simplicity', 0)
    novelty = evaluation.get('novelty', 0)
    scalability = evaluation.get('scalability', 0)
    
    score += viability + value_potential + simplicity + novelty + scalability
    
    if evaluation.get('cringe', 0) == 1:
        score -= 50  # Penalty for cringe
    
    if value_potential < 70 or novelty < 50:
        score = min(score, 300)  # Cap low-impact or common ideas
    
    return max(0, score)

# Archive CSV with timestamp (called only at end)
def archive_csv(csv_file):
    archive_dir = "archive"
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print(f"Created archive directory: {archive_dir}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archived_file = os.path.join(archive_dir, f"idea_log_{timestamp}.csv")
    if os.path.exists(csv_file):
        shutil.copy(csv_file, archived_file)
        print(f"Archived CSV as: {archived_file}")
    else:
        print(f"Warning: {csv_file} not found for archiving.")
    return archived_file

# Main function
def main():
    print("Welcome to the Idea Generator Genetic Algorithm!")
    print("Using LM Studio with Gemma 3 1B.")

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model = "gemma-3-1b"
    evaluate_idea_fn = lambda idea: evaluate_idea(idea, client, model)

    csv_file = "idea_log.csv"
    population_size = 100
    quartile_size = population_size // 4

    print(f"Testing API connectivity with {model}...")
    test_idea = "A Machine Learning-powered system that Analyzes Data in Healthcare"
    evaluation, eval_time = evaluate_idea_fn(test_idea)
    if evaluation is None:
        print("LM Studio connectivity test failed. Check if server is running at http://localhost:1234.")
        return
    print("API connectivity test successful.")

    csv_headers = ["Timestamp", "Generation", "Gene Sequence", "Fitness Score", "Evaluation Time (s)", "Reasoning", "Cringe Attempts"]
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)
            f.flush()

    generation = 0
    population = []  # (idea, evaluation, fitness, cringe_attempts)
    for _ in range(population_size):
        idea = generate_random_idea()
        eval, eval_time = evaluate_idea_fn(idea)
        fitness = calculate_fitness(eval)
        cringe_attempts = 0
        if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
            idea = generate_alternative(idea, client, model)
            eval, eval_time = evaluate_idea_fn(idea)
            fitness = calculate_fitness(eval)
            cringe_attempts += 1
            if eval.get("cringe", 0) == 1 and meets_rewrite_thresholds(eval):
                idea = generate_alternative(idea, client, model, preserve_novelty=True)
                eval, eval_time = evaluate_idea_fn(idea)
                fitness = calculate_fitness(eval)
                cringe_attempts += 1
        
        population.append((idea, eval, fitness, cringe_attempts))
        timestamp = datetime.datetime.now().isoformat()
        reasoning = eval.get('reasoning', 'No reasoning provided') if eval else 'Evaluation failed'
        csv_row = [timestamp, generation, idea, fitness, round(eval_time, 2), reasoning, cringe_attempts]
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)
            f.flush()
            print(f"Wrote row to {csv_file}: {csv_row}")
        time.sleep(0.1)

    try:
        while True:
            print(f"\nGeneration {generation} (Population: {population_size})")
            
            for i, (idea, eval, fitness, cringe_attempts) in enumerate(population):
                original_idea = idea
                eval, eval_time = evaluate_idea_fn(idea)
                fitness = calculate_fitness(eval)
                if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                    idea = generate_alternative(idea, client, model)
                    eval, eval_time = evaluate_idea_fn(idea)
                    fitness = calculate_fitness(eval)
                    cringe_attempts += 1
                    if eval.get("cringe", 0) == 1 and meets_rewrite_thresholds(eval):
                        idea = generate_alternative(idea, client, model, preserve_novelty=True)
                        eval, eval_time = evaluate_idea_fn(idea)
                        fitness = calculate_fitness(eval)
                        cringe_attempts += 1
                population[i] = (idea, eval, fitness, cringe_attempts)
                
                timestamp = datetime.datetime.now().isoformat()
                reasoning = eval.get('reasoning', 'No reasoning provided') if eval else 'Evaluation failed'
                csv_row = [timestamp, generation, idea, fitness, round(eval_time, 2), reasoning, cringe_attempts]
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)
                    f.flush()
                    print(f"Wrote row to {csv_file}: {csv_row}")
                
                time.sleep(0.1)
                
                print(f"Evaluated {i+1}/{population_size}: {idea}")
                print(f"Original Idea: {original_idea}")
                print(f"Final Evaluation: {eval}")
                print(f"Fitness Score: {fitness}")
                print(f"Cringe Attempts: {cringe_attempts}")
                if fitness >= 450:
                    print("*** High Fitness Idea for Coding ***")
                print(f"Evaluation Time: {eval_time:.2f} seconds")

            population.sort(key=lambda x: x[2], reverse=True)
            
            viable_population = [p for p in population if p[3] < 2]  # Max 2 attempts
            q1_top = viable_population[:quartile_size]
            print(f"\nTop Quartile (Q1) - Carried Over (Generation {generation}):")
            for idea, _, score, attempts in q1_top:
                print(f"- {idea} (Score: {score}, Cringe Attempts: {attempts})")

            q2_offspring = []
            for _ in range(quartile_size):
                if len(q1_top) < 2:
                    break
                parent1, parent2 = random.sample(q1_top, 2)
                child = crossover(parent1[0], parent2[0])
                q2_offspring.append((child, None, 0, 0))

            q3_mutations = []
            for _ in range(quartile_size):
                if not q1_top:
                    break
                parent = random.choice(q1_top)
                mutant = mutate(parent[0])
                q3_mutations.append((mutant, None, 0, 0))

            q4_random = []
            for _ in range(quartile_size):
                idea = generate_random_idea()
                eval, eval_time = evaluate_idea_fn(idea)
                fitness = calculate_fitness(eval)
                cringe_attempts = 0
                if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                    idea = generate_alternative(idea, client, model)
                    eval, eval_time = evaluate_idea_fn(idea)
                    fitness = calculate_fitness(eval)
                    cringe_attempts += 1
                    if eval.get("cringe", 0) == 1 and meets_rewrite_thresholds(eval):
                        idea = generate_alternative(idea, client, model, preserve_novelty=True)
                        eval, eval_time = evaluate_idea_fn(idea)
                        fitness = calculate_fitness(eval)
                        cringe_attempts += 1
                q4_random.append((idea, eval, fitness, cringe_attempts))
                timestamp = datetime.datetime.now().isoformat()
                reasoning = eval.get('reasoning', 'No reasoning provided') if eval else 'Evaluation failed'
                csv_row = [timestamp, generation, idea, fitness, round(eval_time, 2), reasoning, cringe_attempts]
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_row)
                    f.flush()
                    print(f"Wrote row to {csv_file}: {csv_row}")
                time.sleep(0.1)

            population = q1_top + q2_offspring + q3_mutations + q4_random
            population = population[:population_size]
            generation += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
        print(f"\nFinal Top Ideas from Generation {generation-1}:")
        population.sort(key=lambda x: x[2], reverse=True)
        for idea, _, score, attempts in population[:quartile_size]:
            print(f"- {idea} (Score: {score}, Cringe Attempts: {attempts})")
        archive_csv(csv_file)
        print("Final CSV archived.")

if __name__ == "__main__":
    main()