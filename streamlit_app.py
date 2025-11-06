import streamlit as st
import pandas as pd
import random
import numpy as np
import io

# ============================================
# 📘 GENETIC ALGORITHM FUNCTIONS
# ============================================

def read_csv_to_dict(df):
    """Convert dataframe to dictionary {Program: [ratings]}"""
    program_ratings = {}
    for _, row in df.iterrows():
        program = row['Type of Program']
        ratings = row.drop('Type of Program').tolist()
        program_ratings[program] = [float(x) for x in ratings]
    return program_ratings


def fitness_function(schedule, ratings_data):
    """Calculate total fitness for a schedule."""
    return sum(
        ratings_data.get(program, [0])[i % len(ratings_data.get(program, [0]))]
        for i, program in enumerate(schedule)
    )


def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]


def crossover(schedule1, schedule2):
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2


def mutate(schedule, all_programs):
    schedule_copy = schedule.copy()
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        mutation_point = random.randint(0, len(schedule_copy) - 1)
        schedule_copy[mutation_point] = random.choice(all_programs)
    return schedule_copy


def add_random_noise_to_ratings(ratings_data, noise_strength=0.05):
    noisy_data = {}
    for k, v in ratings_data.items():
        noisy_data[k] = [max(0, min(1, x + random.uniform(-noise_strength, noise_strength))) for x in v]
    return noisy_data


def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule, best_fitness = None, 0

    for _ in range(generations):
        fitness_scores = [(s, fitness_function(s, ratings_data)) for s in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        best_candidate = fitness_scores[0]

        if best_candidate[1] > best_fitness:
            best_schedule, best_fitness = best_candidate

        new_pop = [x[0] for x in fitness_scores[:elitism_size]]

        while len(new_pop) < population_size:
            p1 = random.choice(fitness_scores[:population_size // 2])[0]
            p2 = random.choice(fitness_scores[:population_size // 2])[0]

            if random.random() < crossover_rate:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            if random.random() < mutation_rate:
                c1 = mutate(c1, all_programs)
            if random.random() < mutation_rate:
                c2 = mutate(c2, all_programs)

            new_pop.extend([c1, c2])

        population = new_pop[:population_size]

    return best_schedule, best_fitness


# ============================================
# 🌐 STREAMLIT APP
# ============================================

st.title("📺 TV Program Scheduling Optimizer (Auto Mode)")

# Try loading your CSV file
file_path = "program_ratings.csv"

try:
    df = pd.read_csv(file_path)
    st.success("✅ Loaded 'program_ratings.csv' successfully.")
except FileNotFoundError:
    st.warning("⚠️ 'program_ratings.csv' not found. Using default embedded dataset.")
    # Default backup dataset (minimal version to ensure app runs)
    csv_data = """Type of Program,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23
live_soccer,0.1,0.1,0.2,0.3,0.5,0.6,0.8,0.9,0.7,0.5,0.4,0.3,0.2,0.1,0.1,0.1,0.1,0.1
reality_show,0.3,0.4,0.5,0.6,0.7,0.8,0.8,0.7,0.6,0.5,0.4,0.3,0.3,0.4,0.5,0.6,0.5,0.4
music_program,0.2,0.2,0.3,0.4,0.4,0.5,0.6,0.6,0.7,0.8,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1
boxing,0.2,0.3,0.4,0.5,0.5,0.5,0.6,0.6,0.5,0.4,0.3,0.3,0.2,0.1,0.1,0.1,0.1,0.1
news,0.3,0.3,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.3"""
    df = pd.read_csv(io.StringIO(csv_data))

st.subheader("📊 Program Ratings Dataset")
st.dataframe(df)

# Run automatically
ratings = read_csv_to_dict(df)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
SCHEDULE_LENGTH = len(all_time_slots)

trials = [
    ("Trial 1", 0.80, 0.20, 0.02, 10),
    ("Trial 2", 0.70, 0.40, 0.05, 20),
    ("Trial 3", 0.60, 0.60, 0.08, 30),
]

for label, co, mu, noise, seed in trials:
    st.header(f"🔹 {label}")
    st.write(f"**Crossover Rate:** {co} | **Mutation Rate:** {mu}")

    random.seed(seed)
    np.random.seed(seed)

    noisy_ratings = add_random_noise_to_ratings(ratings, noise_strength=noise)
    schedule, fitness = genetic_algorithm(
        noisy_ratings, all_programs, SCHEDULE_LENGTH,
        crossover_rate=co, mutation_rate=mu
    )

    df_result = pd.DataFrame({
        "Time Slot": [f"{t:02d}:00" for t in all_time_slots],
        "Program": schedule
    })

    st.dataframe(df_result)
    st.success(f"✅ Best Fitness Score: {fitness:.2f}")
    st.markdown("---")
