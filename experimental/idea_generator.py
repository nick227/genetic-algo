import csv
import time
import random
import datetime
import json
from openai import OpenAI, OpenAIError
from tqdm import tqdm
import argparse
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Seed pools (unchanged)
tech_seeds = ["AI", "Blockchain", "Cloud", "IoT", "Quantum", "Robotics", "AR", "VR", "5G", "Edge", "Microservices", "Serverless", "API", "Database", "Neural", "Biotech", "Nanotech", "Drones", "Wearables", "3DPrinting", "BigData", "Cybersecurity", "DevOps", "MachineLearning", "DeepLearning", "NLP", "ComputerVision", "AugmentedAnalytics", "GraphQL", "Web3", "Metaverse", "DigitalTwin", "ZeroTrust", "SDN", "Containers", "Kubernetes", "FaaS", "SaaS", "PaaS", "IaaS", "NoCode", "LowCode", "RPA", "Chatbots", "Geospatial", "LiDAR", "BlockchainInterop", "SmartContracts", "Tokenization", "EdgeAI", "FederatedLearning", "GANs", "ReinforcementLearning", "QuantumCryptography", "Holography", "Bioinformatics", "SyntheticBiology", "Photovoltaics", "EnergyStorage", "6G", "Terahertz", "SwarmIntelligence", "Neuromorphic", "Exoskeletons", "BrainComputerInterface", "SpatialComputing", "Hyperledger", "DecentralizedID", "MeshNetworks", "AutonomousSystems", "PredictiveMaintenance", "DigitalFabric", "OpticalComputing"]
industry_seeds = ["Health", "Finance", "Education", "Gaming", "Logistics", "Retail", "Energy", "Media", "Travel", "Agritech", "Edtech", "Fintech", "Proptech", "Legal", "Space", "Automotive", "Manufacturing", "Construction", "Hospitality", "Insurance", "Telecom", "Entertainment", "Fashion", "Food", "Pharma", "Biotech", "RealEstate", "Ecommerce", "Marketing", "HR", "Recruitment", "Security", "Defense", "Aerospace", "Maritime", "Mining", "Utilities", "Waste", "Recycling", "Sustainability", "Climate", "Nonprofit", "Government", "PublicSafety", "UrbanPlanning", "SmartCities", "Transportation", "Tourism", "Sports", "Fitness", "Wellness", "ElderCare", "Childcare", "Petcare", "Beauty", "Art", "Music", "Publishing", "Broadcasting", "Advertising", "Events", "Consulting", "Research", "Academia", "Training", "Coaching", "GigEconomy", "RemoteWork", "Crowdsourcing", "SupplyChain", "Warehousing", "Packaging", "Shipping", "Customs", "Forestry", "Fisheries", "Textiles"]
verb_seeds = ["Analyzes", "Optimizes", "Generates", "Secures", "Tracks", "Predicts", "Automates", "Integrates", "Scales", "Monitors", "Visualizes", "Processes", "Deploys", "Encrypts", "Streams", "Aggregates", "Simulates", "Customizes", "Enhances", "Filters", "Distributes", "Synchronizes", "Validates", "Orchestrates", "Transforms", "Extracts", "Classifies", "Detects", "Forecasts", "Personalizes", "Authenticates", "Accelerates", "Balances", "Calibrates", "Compiles", "Coordinates", "Diagnoses", "Digitizes", "Enables", "Evaluates", "Indexes", "Maps", "Merges", "Navigates", "Prioritizes", "Refines", "Replicates", "Routes", "Schedules", "Segments", "Simplifies", "Synthesizes", "Tests", "Trains", "Upgrades", "Verifies", "Adapts", "Amplifies", "Archives", "Boosts", "Captures", "Converts", "Delivers", "Expands", "Facilitates", "Improves", "Links", "Moderates", "Normalizes", "Organizes", "Protects", "Reduces", "Restores", "Shares", "Updates", "Unifies"]
noun_seeds = ["Data", "Platform", "Model", "Network", "System", "Engine", "Framework", "Service", "Application", "Dashboard", "Pipeline", "Cluster", "Interface", "Module", "Sensor", "Algorithm", "Device", "Hub", "Portal", "Tool", "Repository", "Workflow", "Simulation", "Template", "Blueprint", "Map", "Index", "Registry", "Stream", "Layer", "Container", "Node", "Grid", "Chain", "Protocol", "Gateway", "Tracker", "Agent", "Bot", "Widget", "Plugin", "Adapter", "Connector", "Environment", "Suite", "Package", "Stack", "Vault", "Channel", "Feed", "Matrix", "Profile", "Catalog", "Archive", "Library", "Cache", "Queue", "Ledger", "Graph", "Schema", "Instance", "Object", "Asset", "Resource", "Collection", "Sequence", "Array", "Tag", "Signal", "Beam", "Probe", "Lens", "Shield", "Core", "Shell"]
trend_seeds = ["Sustainability", "Decentralization", "Automation", "Personalization", "Gamification", "DigitalTransformation", "CircularEconomy", "RemoteWork", "GigEconomy", "HyperAutomation", "DataPrivacy", "GreenTech", "Resilience", "inclusion", "Diversity", "Equity", "MicroLearning", "Crowdsourcing", "TokenEconomy", "LowTouch", "Contactless", "HybridWork", "Upskilling", "Reskilling", "ClimateAction", "ZeroWaste", "EnergyTransition", "Biohacking", "Mindfulness", "Wellbeing", "SmartLiving", "UrbanMobility", "FoodSecurity", "WaterTech", "HealthEquity", "DigitalNomads", "CreatorEconomy", "SocialImpact", "Transparency", "Traceability", "Regeneration", "OpenSource", "CollaborativeDesign", "VirtualEvents", "ImmersiveExperiences", "OnDemand", "SubscriptionModels", "PlatformEconomy", "AIEthics", "TrustlessSystems", "SelfSovereignty", "MicroTransactions", "NanoInfluencers", "AugmentedWorkforce", "SyntheticMedia", "VoiceFirst", "GestureControl", "SpatialAudio", "HapticFeedback", "Neurotech", "EdgeIntelligence", "QuantumLeap", "PostPandemic", "HyperLocal", "CommunityDriven", "DynamicPricing", "PredictiveHealth", "EcoFriendly", "Minimalism", "FutureProofing"]
user_seeds = ["Consumers", "Businesses", "Students", "Teachers", "Developers", "Entrepreneurs", "Farmers", "Doctors", "Patients", "Travelers", "Gamers", "Retailers", "Manufacturers", "Designers", "Engineers", "Parents", "Kids", "Seniors", "Athletes", "Artists", "Musicians", "Writers", "Researchers", "Scientists", "Investors", "Startups", "Freelancers", "RemoteWorkers", "Nonprofits", "Governments", "Activists", "Volunteers", "Shoppers", "Drivers", "Commuters", "Homeowners", "Renters", "PetOwners", "FitnessEnthusiasts", "Foodies", "EcoConscious", "TechSavvy", "SmallBusinesses", "Enterprises", "Educators", "Trainers", "Coaches", "HealthcareWorkers", "FirstResponders", "LogisticsTeams", "EventPlanners", "Influencers", "Creators", "Streamers", "Collectors", "Hobbyists", "DIYers", "Minimalists", "Nomads", "UrbanDwellers", "RuralCommunities", "Families", "Couples", "Singles", "Retirees", "Veterans", "JobSeekers", "Recruiters", "PolicyMakers", "Advocates", "Innovators"]
benefit_seeds = ["SaveTime", "ReduceCosts", "IncreaseEfficiency", "BoostProductivity", "EnhanceSafety", "ImproveAccuracy", "SimplifyTasks", "GrowRevenue", "StrengthenSecurity", "ExpandAccess", "CutEmissions", "RaiseAwareness", "BuildTrust", "FosterCollaboration", "SpeedDelivery", "LowerRisk", "MaximizeImpact", "OptimizeResources", "EmpowerUsers", "StreamlineOperations", "UnlockInsights", "DriveInnovation", "EnsureCompliance", "EnhanceExperience", "MinimizeWaste", "ScaleEffortlessly", "PersonalizeSolutions", "PreventErrors", "AccelerateGrowth", "SecureData", "ConnectCommunities", "ReduceFriction", "AmplifyReach", "ImproveHealth", "SustainGrowth", "EnableFlexibility", "ProtectPrivacy", "IncreaseEngagement", "SimplifyLearning", "EnhanceMobility", "BoostReliability", "CutDowntime", "PromoteWellness", "StrengthenResilience", "LowerBarriers", "InspireCreativity", "OptimizePerformance", "ExpandOpportunities", "ReduceComplexity", "ImproveDecisionMaking", "SupportSustainability", "EnhanceTransparency", "MinimizeErrors", "GrowCommunity", "IncreaseRetention", "StreamlineAccess", "EmpowerDecisionMakers", "ReduceOverhead", "BoostSatisfaction", "EnableScalability", "ProtectAssets", "ImproveOutcomes", "SimplifyIntegration", "EnhanceUsability", "DriveAdoption", "LowerImpact", "IncreaseClarity", "SupportEquity", "BuildLoyalty", "OptimizeEnergy"]
problem_seeds = ["Inefficiency", "HighCosts", "Complexity", "DataSilos", "UserErrors", "SlowProcesses", "LackOfAccess", "PoorVisibility", "SecurityRisks", "ManualTasks", "LowEngagement", "FragmentedSystems", "Downtime", "ResourceWaste", "SkillGaps", "ScalabilityLimits", "PrivacyConcerns", "Overload", "Inaccuracy", "Disconnection", "TimeDelays", "Redundancy", "LackOfTrust", "PoorUX", "EnergyUse", "AdoptionBarriers", "DataLoss", "Isolation", "ComplianceIssues", "LimitedReach"]
constraint_seeds = ["LowBudget", "NoInternet", "LimitedHardware", "SmallTeam", "ShortTimeline", "LowBandwidth", "MinimalSkills", "NoCloud", "OfflineOnly", "BasicTools", "SingleDevice", "LowPower", "FreeTier", "OpenSourceOnly", "OnePerson", "QuickBuild", "NoFunding", "LocalOnly", "SimpleUI", "NoAPI", "Lightweight", "NoServers", "ManualSetup", "TinyFootprint", "ZeroCost"]
context_seeds = ["UrbanAreas", "RuralAreas", "Emergencies", "Workplaces", "Homes", "Classrooms", "Factories", "Fields", "Hospitals", "RetailStores", "PublicSpaces", "Vehicles", "Events", "Disasters", "Travel", "Offices", "RemoteLocations", "Communities", "Outdoors", "Indoors", "MobileUse", "Nighttime", "PeakHours", "LowSignal", "CrowdedPlaces", "QuietZones", "ExtremeWeather", "DevelopingRegions", "HighTraffic", "TemporarySetup"]
emotion_seeds = ["Delight", "Calm", "Motivate", "Reassure", "Excite", "Simplify", "Empower", "Inspire", "Comfort", "Engage", "Satisfy", "Trust", "Amuse", "Focus", "Relieve", "Energize", "Connect", "Surprise", "Encourage", "Clarify", "Reward", "Ease", "Thrill", "Support", "Uplift", "Unify", "Intrigue", "Relax", "Validate", "Celebrate"]

# Error Codes
ERROR_TIMEOUT = "E001"
ERROR_CONNECTION = "E002"
ERROR_JSON = "E003"
ERROR_UNKNOWN = "E999"

def generate_random_idea(tech, verbs, nouns, industries, trends, users, benefits, problems, constraints, contexts, emotions, topic=None):
    parts = []
    parts.append(f"{random.choice(verbs)} {random.choice(nouns)} with {random.choice(tech)} ")
    extra_parts = []
    if random.random() < 0.5:
        extra_parts.append(f"for {random.choice(industries)}")
    if random.random() < 0.5:
        extra_parts.append(f"solving {random.choice(problems)}")
    if random.random() < 0.5:
        extra_parts.append(f"in {random.choice(contexts)}")
    if random.random() < 0.5:
        extra_parts.append(f"to {random.choice(benefits)} {random.choice(users)}")
    if random.random() < 0.5:
        extra_parts.append(f"under {random.choice(constraints)}")
    if random.random() < 0.5:
        extra_parts.append(f"leveraging {random.choice(trends)}")
    if random.random() < 0.5:
        extra_parts.append(f"to {random.choice(emotions)} users")
    if topic and random.random() < 0.7:
        extra_parts.append(f"focused on {topic}")
    if extra_parts:
        if len(extra_parts) > 1:
            parts.append(", ".join(extra_parts))
        else:
            parts.append(extra_parts[0])
    return " ".join(parts).strip()

def crossover(idea1, idea2, tech, verbs, nouns, industries, trends, users, benefits, problems, constraints, contexts, emotions, topic=None):
    parts1 = idea1.split()
    parts2 = idea2.split()
    base = f"{parts1[0]} {random.choice(nouns)} with {random.choice(tech)}"
    if len(parts2) > 3 and random.random() < 0.5:
        base += f" {parts2[3]} {parts2[4]}"
    return base

def mutate(idea, rate, tech, verbs, nouns, industries, trends, users, benefits, problems, constraints, contexts, emotions, topic=None):
    if random.random() < rate:
        parts = idea.split()
        return f"{random.choice(verbs)} {parts[1]} with {random.choice(tech)}"
    return idea

def generate_alternative(idea, client, model, timeout=60):
    prompt = f"""Given the idea: '{idea}', rewrite it into a slightly longer, yet still concise and clear version. Add one or two meaningful words to enhance clarity or specificity while keeping it short and avoiding fluff. Return only the rewritten idea as a string."""
    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=50,
            timeout=timeout
        )
        return completion.choices[0].message.content.strip(), time.time() - start, None
    except OpenAIError as e:
        if "timeout" in str(e).lower():
            logging.error(f"Timeout in generate_alternative for '{idea}': {str(e)}")
            return idea, time.time() - start, f"{ERROR_TIMEOUT}: {str(e)}"
        elif "connection" in str(e).lower():
            logging.error(f"Connection error in generate_alternative for '{idea}': {str(e)}")
            return idea, time.time() - start, f"{ERROR_CONNECTION}: {str(e)}"
        else:
            logging.error(f"Unknown error in generate_alternative for '{idea}': {str(e)}")
            return idea, time.time() - start, f"{ERROR_UNKNOWN}: {str(e)}"

def evaluate_idea(idea, client, model, timeout=60):
    prompt = f"""You are a pragmatic, no-nonsense tech strategist with a razor-sharp eye for detail and zero tolerance for fluff. Evaluate the idea: '{idea}' as a standalone concept, ignoring any prior ideas or context. Assess it independently and accurately based on these exact keys with integer ranges:
- "viability": 0-100 (technical feasibility and real-world execution; proven concepts >50, plausible but untested 30-50, impractical <30)
- "value_potential": 0-100 (market demand and revenue potential; broad, tangible appeal >50, niche or vague <40)
- "simplicity": 0-100 (ease of implementation; straightforward >50, complex or unclear <30)
- "novelty": 0-100 (innovation level; genuinely fresh >50, common or derivative <40)
- "scalability": 0-100 (growth potential across markets; wide reach >50, limited scope <40)
- "small_team_feasibility": 0-100 (feasibility for 2-3 people with limited resources to build cheaply; low complexity, basic skills, minimal infrastructure, quick MVP >50; high complexity or cost <30)
- "cringe": 0 or 1 (1 if grammatically broken, logically incoherent, or nonsensical—like 'Visualizes Sensor'; 0 if structurally sound, even if bold)
- "reasoning": a string (concisely explain the idea and justify scores; focus on facts, vary phrasing, no fluff)
Return only the JSON object as a string. Be ruthless with nonsense, precise with scores, and treat each idea as a fresh evaluation."""
    start = time.time()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400,
            timeout=timeout
        )
        content = completion.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3]
        eval = json.loads(content)
        for key in ['viability', 'value_potential', 'simplicity', 'novelty', 'scalability', 'small_team_feasibility', 'cringe']:
            eval[key] = int(eval[key])
        return eval, time.time() - start, None
    except OpenAIError as e:
        if "timeout" in str(e).lower():
            logging.error(f"Timeout in evaluate_idea for '{idea}': {str(e)}")
            error_code = ERROR_TIMEOUT
        elif "connection" in str(e).lower():
            logging.error(f"Connection error in evaluate_idea for '{idea}': {str(e)}")
            error_code = ERROR_CONNECTION
        else:
            logging.error(f"Unknown error in evaluate_idea for '{idea}': {str(e)}")
            error_code = ERROR_UNKNOWN
        return {"viability": 0, "value_potential": 0, "simplicity": 0, "novelty": 0, "scalability": 0, "small_team_feasibility": 0, "cringe": 1, "reasoning": str(e)}, time.time() - start, f"{error_code}: {str(e)}"
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing error in evaluate_idea for '{idea}': {str(e)}")
        return {"viability": 0, "value_potential": 0, "simplicity": 0, "novelty": 0, "scalability": 0, "small_team_feasibility": 0, "cringe": 1, "reasoning": "Invalid JSON response"}, time.time() - start, f"{ERROR_JSON}: {str(e)}"

def calculate_fitness(eval, generation, max_generations):
    weights = {
        "viability": 2.0,
        "novelty": 2.0 + (generation / max_generations) * 0.5,
        "value_potential": 1.0,
        "simplicity": 0.8,
        "scalability": 0.8,
        "small_team_feasibility": 10.0
    }
    score = sum(int(eval.get(k, 0)) * weights[k] for k in weights)
    if eval.get("cringe", 0) == 1:
        score -= min(50, score * 0.2)
    return max(0, score)

def log_idea(csv_file, generation, idea, eval, fitness, eval_time, cringe_attempts, error=None):
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.datetime.now().isoformat(), generation, idea, fitness,
                         eval.get('viability', 0), eval.get('value_potential', 0), eval.get('simplicity', 0),
                         eval.get('novelty', 0), eval.get('scalability', 0), eval.get('small_team_feasibility', 0), eval.get('cringe', 0),
                         round(eval_time, 2), eval.get('reasoning', ''), cringe_attempts, error or ''])
        f.flush()

def initialize_csv(csv_file):
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Generation", "Gene Sequence", "Fitness Score", "Viability",
                             "Value Potential", "Simplicity", "Novelty", "Scalability", "Small Team Feasibility", "Cringe",
                             "Evaluation Time (s)", "Reasoning", "Cringe Attempts", "Error"])

def load_previous_population(csv_file, target_population_size, client, model, max_generations):
    if not os.path.exists(csv_file):
        logging.info(f"No existing CSV found at {csv_file}. Starting fresh.")
        return 0, [], set()

    all_ideas = []
    seen_ideas = set()
    last_generation = -1

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = int(row["Generation"])
            last_generation = max(last_generation, gen)
            eval_dict = {
                "viability": int(row["Viability"]),
                "value_potential": int(row["Value Potential"]),
                "simplicity": int(row["Simplicity"]),
                "novelty": int(row["Novelty"]),
                "scalability": int(row["Scalability"]),
                "small_team_feasibility": int(row["Small Team Feasibility"]),
                "cringe": int(row["Cringe"]),
                "reasoning": row["Reasoning"]
            }
            fitness = float(row["Fitness Score"])
            idea_tuple = (row["Gene Sequence"], eval_dict, fitness, int(row["Cringe Attempts"]), float(row["Evaluation Time (s)"]), row["Error"] or None)
            all_ideas.append((gen, idea_tuple))
            seen_ideas.add(row["Gene Sequence"])

    if not all_ideas:
        logging.info(f"CSV {csv_file} is empty. Starting fresh.")
        return 0, [], set()

    # Sort by generation (descending) and fitness within each generation
    all_ideas.sort(key=lambda x: (-x[0], -x[1][2]))  # -x[0] for latest gen first, -x[1][2] for highest fitness

    # Infer CSV population size from the last generation
    last_gen_ideas = [idea for gen, idea in all_ideas if gen == last_generation]
    csv_population_size = len(last_gen_ideas)

    # Select population based on target_population_size
    initial_population = []
    if target_population_size <= csv_population_size:
        # Take top performers from the last generation
        initial_population = [idea for _, idea in all_ideas if _ == last_generation][:target_population_size]
        logging.info(f"Target population_size ({target_population_size}) <= CSV ({csv_population_size}). Took top {target_population_size} from gen {last_generation}.")
    else:
        # Use most recent generations, fill remaining with random later
        remaining = target_population_size
        for gen, idea in all_ideas:
            if remaining <= 0:
                break
            initial_population.append(idea)
            remaining -= 1
        logging.info(f"Target population_size ({target_population_size}) > CSV ({csv_population_size}). Loaded {len(initial_population)} from recent generations.")

    return last_generation + 1, initial_population, seen_ideas

def main():
    parser = argparse.ArgumentParser(description="Idea Generator GA")
    parser.add_argument('--population_size', type=int, default=4, help="Number of ideas per generation")
    parser.add_argument('--max_generations', type=int, default=50, help="Number of generations")
    parser.add_argument('--topic', type=str, default=None, help="Optional topic for all ideas (e.g., 'sustainability')")
    parser.add_argument('--csv_file', type=str, default=None, help="Path to existing idea_log CSV to resume from")
    args = parser.parse_args()

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    model = "gemma-3-1b"
    csv_file = args.csv_file if args.csv_file else f"idea_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    initialize_csv(csv_file)

    # Load previous population
    start_generation, initial_population, seen_ideas = load_previous_population(csv_file, args.population_size, client, model, args.max_generations)
    random.seed(datetime.datetime.now().timestamp())

    try:
        for generation in range(start_generation, args.max_generations + 1):
            mutation_rate = 0.1 + (generation / args.max_generations) * 0.2
            logging.info(f"Starting Generation {generation} (Mutation Rate: {mutation_rate:.2f})")

            available_tech = tech_seeds.copy()
            available_verbs = verb_seeds.copy()
            available_nouns = noun_seeds.copy()
            available_industries = industry_seeds.copy()
            available_trends = trend_seeds.copy()
            available_users = user_seeds.copy()
            available_benefits = benefit_seeds.copy()
            available_problems = problem_seeds.copy()
            available_constraints = constraint_seeds.copy()
            available_contexts = context_seeds.copy()
            available_emotions = emotion_seeds.copy()

            if generation == start_generation and initial_population:
                population = initial_population
                if len(population) < args.population_size:
                    remaining = args.population_size - len(population)
                    logging.info(f"Population loaded ({len(population)}) < target ({args.population_size}). Generating {remaining} random ideas.")
                    with tqdm(total=remaining, desc=f"Filling Gen {generation}") as pbar:
                        while len(population) < args.population_size:
                            idea = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries,
                                                       available_trends, available_users, available_benefits, available_problems,
                                                       available_constraints, available_contexts, available_emotions, args.topic)
                            if idea not in seen_ideas:
                                seen_ideas.add(idea)
                                eval, eval_time, error = evaluate_idea(idea, client, model)
                                fitness = calculate_fitness(eval, generation, args.max_generations)
                                population.append((idea, eval, fitness, 0, eval_time, error))
                                log_idea(csv_file, generation, idea, eval, fitness, eval_time, 0, error)
                                pbar.update(1)
            elif generation == 0:
                population = []
                with tqdm(total=args.population_size, desc=f"Generation {generation}") as pbar:
                    while len(population) < args.population_size:
                        idea = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries,
                                                   available_trends, available_users, available_benefits, available_problems,
                                                   available_constraints, available_contexts, available_emotions, args.topic)
                        if idea not in seen_ideas:
                            seen_ideas.add(idea)
                            eval, eval_time, error = evaluate_idea(idea, client, model)
                            total_eval_time = eval_time
                            fitness = calculate_fitness(eval, generation, args.max_generations)
                            cringe_attempts = 0
                            if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                logging.warning(f"Cringe flagged: {idea} -> {eval['reasoning']}")
                                new_idea, alt_time, alt_error = generate_alternative(idea, client, model)
                                total_eval_time += alt_time
                                if new_idea not in seen_ideas:
                                    logging.info(f"Rewritten to: {new_idea}")
                                    idea = new_idea
                                    eval, eval_time, error = evaluate_idea(idea, client, model)
                                    total_eval_time += eval_time
                                    fitness = calculate_fitness(eval, generation, args.max_generations)
                                    seen_ideas.add(idea)
                                    cringe_attempts += 1
                                if alt_error:
                                    error = alt_error if not error else f"{error}; {alt_error}"
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
                            child = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries,
                                                        available_trends, available_users, available_benefits, available_problems,
                                                        available_constraints, available_contexts, available_emotions, args.topic)
                            if child not in seen_ideas:
                                seen_ideas.add(child)
                                eval, eval_time, error = evaluate_idea(child, client, model)
                                total_eval_time = eval_time
                                fitness = calculate_fitness(eval, generation, args.max_generations)
                                cringe_attempts = 0
                                if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                    logging.warning(f"Cringe flagged: {child} -> {eval['reasoning']}")
                                    new_child, alt_time, alt_error = generate_alternative(child, client, model)
                                    total_eval_time += alt_time
                                    if new_child not in seen_ideas:
                                        logging.info(f"Rewritten to: {new_child}")
                                        child = new_child
                                        eval, eval_time, error = evaluate_idea(child, client, model)
                                        total_eval_time += eval_time
                                        fitness = calculate_fitness(eval, generation, args.max_generations)
                                        seen_ideas.add(child)
                                        cringe_attempts += 1
                                    if alt_error:
                                        error = alt_error if not error else f"{error}; {alt_error}"
                                next_population.append((child, eval, fitness, cringe_attempts, total_eval_time, error))
                                log_idea(csv_file, generation, child, eval, fitness, total_eval_time, cringe_attempts, error)
                                pbar.update(1)

                    for _ in range(mutation_size):
                        if len(next_population) >= args.population_size:
                            break
                        if viable:
                            mutant = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries,
                                                         available_trends, available_users, available_benefits, available_problems,
                                                         available_constraints, available_contexts, available_emotions, args.topic)
                            if mutant not in seen_ideas:
                                seen_ideas.add(mutant)
                                eval, eval_time, error = evaluate_idea(mutant, client, model)
                                total_eval_time = eval_time
                                fitness = calculate_fitness(eval, generation, args.max_generations)
                                cringe_attempts = 0
                                if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                    logging.warning(f"Cringe flagged: {mutant} -> {eval['reasoning']}")
                                    new_mutant, alt_time, alt_error = generate_alternative(mutant, client, model)
                                    total_eval_time += alt_time
                                    if new_mutant not in seen_ideas:
                                        logging.info(f"Rewritten to: {new_mutant}")
                                        mutant = new_mutant
                                        eval, eval_time, error = evaluate_idea(mutant, client, model)
                                        total_eval_time += eval_time
                                        fitness = calculate_fitness(eval, generation, args.max_generations)
                                        seen_ideas.add(mutant)
                                        cringe_attempts += 1
                                    if alt_error:
                                        error = alt_error if not error else f"{error}; {alt_error}"
                                next_population.append((mutant, eval, fitness, cringe_attempts, total_eval_time, error))
                                log_idea(csv_file, generation, mutant, eval, fitness, total_eval_time, cringe_attempts, error)
                                pbar.update(1)

                    for _ in range(random_size):
                        if len(next_population) >= args.population_size:
                            break
                        idea = generate_random_idea(available_tech, available_verbs, available_nouns, available_industries,
                                                   available_trends, available_users, available_benefits, available_problems,
                                                   available_constraints, available_contexts, available_emotions, args.topic)
                        if idea not in seen_ideas:
                            seen_ideas.add(idea)
                            eval, eval_time, error = evaluate_idea(idea, client, model)
                            total_eval_time = eval_time
                            fitness = calculate_fitness(eval, generation, args.max_generations)
                            cringe_attempts = 0
                            if eval.get("cringe", 0) == 1 and cringe_attempts < 2:
                                logging.warning(f"Cringe flagged: {idea} -> {eval['reasoning']}")
                                new_idea, alt_time, alt_error = generate_alternative(idea, client, model)
                                total_eval_time += alt_time
                                if new_idea not in seen_ideas:
                                    logging.info(f"Rewritten to: {new_idea}")
                                    idea = new_idea
                                    eval, eval_time, error = evaluate_idea(idea, client, model)
                                    total_eval_time += eval_time
                                    fitness = calculate_fitness(eval, generation, args.max_generations)
                                    seen_ideas.add(idea)
                                    cringe_attempts += 1
                                if alt_error:
                                    error = alt_error if not error else f"{error}; {alt_error}"
                            next_population.append((idea, eval, fitness, cringe_attempts, total_eval_time, error))
                            log_idea(csv_file, generation, idea, eval, fitness, total_eval_time, cringe_attempts, error)
                            pbar.update(1)

                population = next_population[:args.population_size]

            avg_fitness = sum(p[2] for p in population) / len(population) if population else 0
            best_fitness = max(p[2] for p in population) if population else 0
            logging.info(f"Generation {generation} completed. Average Fitness: {avg_fitness:.2f} | Best Fitness: {best_fitness:.2f}")

    except KeyboardInterrupt:
        logging.info("Program interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {str(e)}")

if __name__ == "__main__":
    main()