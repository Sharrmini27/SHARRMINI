import streamlit as st
import pandas as pd
import random
import numpy as np
import io

# ============================================
# 📘 EMBEDDED CSV DATA (from your uploaded file)
# ============================================

csv_data = """Type of Program,Hour 6,Hour 7,Hour 8,Hour 9,Hour 10,Hour 11,Hour 12,Hour 13,Hour 14,Hour 15,Hour 16,Hour 17,Hour 18,Hour 19,Hour 20,Hour 21,Hour 22,Hour 23
news,0.1,0.1,0.4,0.3,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.5,0.3,0.4,0.3,0.2,0.1,0.2
live_soccer,0.0,0.1,0.0,0.2,0.1,0.3,0.2,0.1,0.4,0.3,0.4,0.5,0.4,0.6,0.4,0.3,0.4,0.3
movie_a,0.1,0.1,0.2,0.4,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.3,0.5,0.3,0.4
movie_b,0.2,0.1,0.1,0.3,0.2,0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.4,0.5,0.4,0.3,0.4,0.5
reality_show,0.3,0.4,0.3,0.4,0.4,0.4,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.2,0.3,0.2,0.2,0.3
"""

df = pd.read_csv(io.StringIO(csv_data))

# ============================================
# 📘 GENETIC ALGORITHM ENGINE
# ============================================

def read_csv_to_dict(df):
    program_ratings = {}
    for _, row in df.iterrows():
        program = row["Type of Program"]
        ratings = row.drop("Type of Program").tolist()
        program_ratings[program] = [float(x) for x in ratings]
    return program_ratings


def fitness_function(schedule, ratings_data):
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


def mutate(schedule, all_programs, mutation_strength=0.1):
    schedule_copy = schedule.copy()
    num_mutations = random.randint(1, 4)
    for _ in range(num_mutations):
        mutation_point = random.randint(0, len(schedule_copy) - 1)
        schedule_copy[mutation_point] = random.choice(all_programs)
    return schedule_copy


def add_random_noise_to_ratings(ratings_data, noise_strength=0.05):
    noisy_data = {}
    for k, v in ratings_data.items():
        noisy_data[k] = [max(0, min(5, x + random.uniform(-noise_strength, noise_strength))) for x in v]
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
# 📺 STREAMLIT FRONT-END
# ============================================

st.title("📺 Genetic Algorithm — TV Program Scheduling Optimizer (Auto Mode)")

st.success("✅ Embedded dataset loaded successfully — no upload required.")

st.subheader("📊 Program Ratings Dataset")
st.dataframe(df)

ratings = read_csv_to_dict(df)
all_programs = list(ratings.keys())
all_time_slots = list(range(6, 24))
SCHEDULE_LENGTH = len(all_time_slots)

trials = [
    ("Trial 1", 0.8, 0.2, 0.02, 10),
    ("Trial 2", 0.7, 0.4, 0.05, 20),
    ("Trial 3", 0.6, 0.6, 0.08, 30),
]

# Auto-run all three trials
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
