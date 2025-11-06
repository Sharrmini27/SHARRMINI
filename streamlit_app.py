import streamlit as st
import pandas as pd
import random
import numpy as np
import io

# ============================================
# 📘 GENETIC ALGORITHM ENGINE
# ============================================

def read_csv_to_dict(uploaded_file):
    """Reads uploaded CSV (from Streamlit uploader) and returns {Program: [ratings]}"""
    program_ratings = {}
    try:
        uploaded_file.seek(0)  # Reset pointer to start
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))

        for _, row in df.iterrows():
            program = row['Type of Program']
            ratings = row.drop('Type of Program').tolist()
            program_ratings[program] = [float(x) for x in ratings]

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
    return program_ratings


def fitness_function(schedule, ratings_data):
    """Calculates total fitness for a schedule."""
    return sum(
        ratings_data.get(program, [0])[i % len(ratings_data.get(program, [0]))]
        for i, program in enumerate(schedule)
    )


def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]


def crossover(schedule1, schedule2):
    """Single-point crossover."""
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    point = random.randint(1, len(schedule1) - 1)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2


def mutate(schedule, all_programs):
    """Mutates multiple random genes."""
    schedule_copy = schedule.copy()
    num_mutations = random.randint(1, 3)
    for _ in range(num_mutations):
        mutation_point = random.randint(0, len(schedule_copy) - 1)
        schedule_copy[mutation_point] = random.choice(all_programs)
    return schedule_copy


def add_random_noise_to_ratings(ratings_data, noise_strength=0.05):
    """Adds small noise to ratings for variety between trials."""
    noisy_data = {}
    for k, v in ratings_data.items():
        noisy_data[k] = [max(0, min(1, x + random.uniform(-noise_strength, noise_strength))) for x in v]
    return noisy_data


def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    """Core GA loop."""
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

st.title("📺 Genetic Algorithm — TV Program Scheduling Optimizer")

st.info("""
### 🧾 Instructions
1. Upload your **`program_ratings.csv`** file using the uploader below.  
2. The file must have:
   - First column: **Type of Program**
   - Next columns: **Hour 6** to **Hour 23**
3. Click **“🚀 Run All 3 Trials”** to compare different GA configurations.
""")

uploaded_file = st.file_uploader("📂 Upload your program_ratings.csv", type=["csv"])

if uploaded_file is not None:
    # ✅ Read dataset safely
    uploaded_file.seek(0)
    df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode("utf-8")))
    st.subheader("📊 Program Ratings Dataset")
    st.dataframe(df)

    ratings = read_csv_to_dict(uploaded_file)

    if ratings:
        all_programs = list(ratings.keys())
        all_time_slots = list(range(6, 24))
        SCHEDULE_LENGTH = len(all_time_slots)

        # 3 experimental trials
        trials = [
            ("Trial 1", 0.85, 0.25, 0.02, 10),
            ("Trial 2", 0.70, 0.45, 0.05, 20),
            ("Trial 3", 0.60, 0.60, 0.08, 30),
        ]

        if st.button("🚀 Run All 3 Trials"):
            for label, co, mu, noise, seed in trials:
                st.header(f"🔹 {label}")
                st.write(f"**Crossover Rate:** {co} | **Mutation Rate:** {mu}")

                random.seed(seed)
                np.random.seed(seed)

                # Add random noise for variation between trials
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
                st.success(f"✅ Best Fitness Score: {fitness:.4f}")
                st.markdown("---")
    else:
        st.error("⚠️ Could not read program ratings correctly.")
else:
    st.warning("📂 Please upload your CSV file to begin.")
