import re
import os
import sys
import json
from openai import OpenAI
from datetime import datetime

# Initialize LM Studio client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
MODEL = "gemma-3-1b"
DEFAULT_MAX_DEPTH = 3  # Max sub-step layers below main (total depth 4)
MAX_BREADTH = 3  # Fixed breadth for depth 1 and beyond

def stream_model_response(prompt):
    """Stream the LLM's response and return the full text."""
    full_response = ""
    print(f"\nStreaming tokens for prompt: {prompt[:50]}...")
    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                token = chunk.choices[0].delta.content
                print(token, end='', flush=True)
                full_response += token
        print("\nStreaming complete.\n")
        return full_response.strip()
    except Exception as e:
        print(f"\nError streaming from LLM: {e}\n")
        return ""

def parse_steps(response):
    """Parse LLM response as JSON into (step_name, description) tuples."""
    print(f"Raw LLM response:\n{response}")
    json_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', response, re.DOTALL)
    if not json_match:
        print("No valid JSON array found in response.")
        return []
    json_str = json_match.group(0)
    json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing comma
    try:
        steps_data = json.loads(json_str)
        if not isinstance(steps_data, list):
            print("JSON is not a list of steps.")
            return []
        steps = []
        seen_names = set()
        for step in steps_data:
            if not isinstance(step, dict) or "step_name" not in step or "description" not in step:
                print(f"Skipping invalid step format: {step}")
                continue
            step_name = step["step_name"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "").replace("–", "_").replace(":", "_")
            description = step["description"].strip()
            if not step_name or step_name in ('steps', '[steps]'):
                print(f"Skipping invalid step name: {step_name}")
                continue
            if step_name not in seen_names:
                steps.append((step_name, description))
                seen_names.add(step_name)
                print(f"Parsed: {step_name} - {description}")
        if not steps:
            print("No valid steps parsed from JSON.")
        return steps
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error parsing response: {e}")
        return []

def update_output_file(filename, tree_dict):
    """Write the full tree to the file with tick marks and aligned comments."""
    try:
        max_name_length = max(len(name) for name in tree_dict.keys())
        column_width = max_name_length + 12  # 12 spaces (2 tabs + original buffer)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Task Breakdown for: {overall_description}\n")
            f.write(f"# Updated: {datetime.now().isoformat()}\n\n")
            f.write("# Tree\n")
            
            def write_tree(node_name, prefix="", is_last=False):
                node = tree_dict[node_name]
                if node_name == "main":
                    line = f"{node_name:<{column_width}} # {node['description']}"
                    f.write(f"{line}\n")
                else:
                    tick = '└── ' if is_last else '├── '
                    name_with_tick = f"{prefix}{tick}{node_name}"
                    padding = " " * (column_width - len(name_with_tick))
                    line = f"{name_with_tick}{padding} # {node['description']}"
                    f.write(f"{line}\n")
                children = sorted(node["children"])
                for i, child in enumerate(children):
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    write_tree(child, new_prefix, i == len(children) - 1)
            
            write_tree("main")
            f.flush()
            os.fsync(f.fileno())
        print(f"File {filename} updated successfully.")
    except Exception as e:
        print(f"Error writing to file: {e}")

def generate_level_steps(parent_name, description, overall_description, depth):
    """Generate steps for a single parent at the given depth."""
    breadth_limit = "any number of" if depth == 0 else f"up to {MAX_BREADTH}"
    prompt = (
        f"For the goal '{overall_description}', break down '{parent_name}' described as '{description}' "
        f"into {breadth_limit} smaller, practical steps. Do not include code—just list unique step names and short descriptions. "
        f"Each step must be a single, realistic action directly relevant to '{overall_description}'. "
        f"Use clear, simple language for non-technical tasks (e.g., 'prepare_item: get the necessary materials') "
        f"and precise technical terms for technical tasks (e.g., 'initialize_component: set up the core system'). "
        f"If '{description}' is a single action, return it unchanged. Ensure steps are distinct, avoid repetition, "
        f"and tailor them specifically to the provided goal. "
        f"Return the response in JSON format as an array of objects, each with 'step_name' and 'description' keys, "
        f"e.g., [{{\"step_name\": \"start_task\", \"description\": \"Begin the main activity\"}}, ...]."
    )
    response = stream_model_response(prompt)
    return parse_steps(response)

def revise_steps(tree_dict, overall_description):
    """Pass the entire tree to the LLM for each item to revise for consistency."""
    revised_tree = tree_dict.copy()
    
    def build_tree_text(node_name, prefix=""):
        node = tree_dict[node_name]
        lines = [f"{prefix}{node_name}  # {node['description']}"]
        for child in sorted(node["children"]):
            lines.extend(build_tree_text(child, prefix + "  "))
        return lines
    
    full_tree_text = "\n".join(build_tree_text("main"))
    
    for step_name in tree_dict:
        if step_name == "main":
            continue  # Skip revising 'main'
        original_desc = tree_dict[step_name]["description"]
        parent_name = tree_dict[step_name]["parent"]
        prompt = (
            f"Given the goal '{overall_description}', here is the full task breakdown tree:\n\n"
            f"{full_tree_text}\n\n"
            f"Review the step '{step_name}' described as '{original_desc}' under '{parent_name}'. "
            f"Revise its description to ensure it is consistent with the overall tree structure, "
            f"clearly defined, and directly relevant to the goal. Return a single revised step in JSON format: "
            f"[{{\"step_name\": \"{step_name}\", \"description\": \"revised description\"}}]."
        )
        response = stream_model_response(prompt)
        revised_steps = parse_steps(response)
        if revised_steps and revised_steps[0][0] == step_name.split("_")[0]:  # Match base name
            revised_tree[step_name]["description"] = revised_steps[0][1]
            print(f"Revised: {step_name} - {revised_steps[0][1]}")
    
    return revised_tree

def generate_program(max_depth=DEFAULT_MAX_DEPTH, max_breadth=MAX_BREADTH, goal=None):
    """Generate a task breakdown, writing all main tasks first, then substeps in real time, then revise."""
    global overall_description
    if goal:
        overall_description = goal
    else:
        overall_description = input("Enter the project goal (e.g., 'create a data pipeline'): ")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"task_breakdown_{timestamp}.txt"
    tree_dict = {"main": {"description": f"start and complete the goal '{overall_description}'", "children": [], "parent": None}}
    defined_functions = set()
    
    # Write initial 'main' to file
    update_output_file(output_filename, tree_dict)
    print(f"Created: {output_filename}")
    
    # Generate and write all depth 0 tasks first
    print("\nGenerating all main tasks (depth 0)")
    main_steps = generate_level_steps("main", tree_dict["main"]["description"], overall_description, 0)
    if not main_steps:
        print("No valid main tasks generated. Exiting.")
        return
    
    defined_functions.add("main")
    next_level = []
    for step_name, description in main_steps:
        unique_step_name = f"{step_name}_main"
        if unique_step_name not in tree_dict:
            tree_dict[unique_step_name] = {"description": description, "children": [], "parent": "main"}
            tree_dict["main"]["children"].append(unique_step_name)
            next_level.append((unique_step_name, description))
            print(f"Added main task: {unique_step_name}")
    
    # Write all main tasks to file before proceeding
    update_output_file(output_filename, tree_dict)
    print("All main tasks written to file. Proceeding to substeps.")
    
    # Now generate substeps in real time for depths 1 and beyond
    current_level = next_level
    depth = 1
    
    while current_level and depth <= max_depth:
        print(f"\nGenerating steps for depth {depth}")
        next_level = []
        for parent_name, parent_desc in current_level:
            if parent_name not in defined_functions:
                steps = generate_level_steps(parent_name, parent_desc, overall_description, depth)
                if not steps:
                    print(f"No valid steps for {parent_name}. Atomic.")
                    steps = [(parent_name, parent_desc)]
                if len(steps) == 1 and steps[0][0] == parent_name and steps[0][1] == parent_desc:
                    defined_functions.add(parent_name)
                else:
                    defined_functions.add(parent_name)
                    steps = steps[:MAX_BREADTH]  # Limit sub-steps to MAX_BREADTH
                    for step_name, description in steps:
                        unique_step_name = f"{step_name}_{parent_name}"
                        if unique_step_name not in tree_dict:
                            tree_dict[unique_step_name] = {"description": description, "children": [], "parent": parent_name}
                            tree_dict[parent_name]["children"].append(unique_step_name)
                            next_level.append((unique_step_name, description))
                            print(f"Added: {unique_step_name} under {parent_name}")
                            update_output_file(output_filename, tree_dict)
                        else:
                            print(f"Skipping duplicate step: {unique_step_name}")
        
        if not next_level:
            print(f"No new steps generated at depth {depth}. Stopping.")
            break
        current_level = next_level
        depth += 1
    
    # Final revision step
    print("\nRevising the entire tree for consistency...")
    revised_tree = revise_steps(tree_dict, overall_description)
    update_output_file(output_filename, revised_tree)
    print(f"\nDone. Final revised tree in {output_filename}")

if __name__ == "__main__":
    print("Task Breakdown Generator")
    max_depth = DEFAULT_MAX_DEPTH
    max_breadth = MAX_BREADTH
    goal = None
    
    if len(sys.argv) > 1:
        if len(sys.argv) == 2:
            goal = sys.argv[1]
        elif len(sys.argv) == 3:
            try:
                max_depth = int(sys.argv[1])
                max_breadth = int(sys.argv[2])
                if max_depth < 1 or max_breadth < 1:
                    raise ValueError("Depth and breadth must be positive.")
            except ValueError as e:
                print(f"Error: {e}. Using defaults: max_depth={DEFAULT_MAX_DEPTH}, max_breadth={MAX_BREADTH}")
        elif len(sys.argv) == 4:
            try:
                max_depth = int(sys.argv[1])
                max_breadth = int(sys.argv[2])
                if max_depth < 1 or max_breadth < 1:
                    raise ValueError("Depth and breadth must be positive.")
                goal = sys.argv[3]
            except ValueError as e:
                print(f"Error: {e}. Using defaults: max_depth={DEFAULT_MAX_DEPTH}, max_breadth={MAX_BREADTH}")
        else:
            print("Usage: python script.py [max_depth max_breadth] [goal]")
            print(f"Defaults: max_depth={DEFAULT_MAX_DEPTH}, max_breadth={MAX_BREADTH}")
            sys.exit(1)
    
    print(f"Running with max_depth={max_depth}, max_breadth={max_breadth}")
    generate_program(max_depth, max_breadth, goal)