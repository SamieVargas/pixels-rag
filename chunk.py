"""Convert a single day's row dict into a human-readable text chunk for embedding.

This is the most important file for retrieval quality — the chunk text is what
gets embedded, so it needs to be information-dense.
"""


def row_to_chunk(row: dict) -> str:
    """
    Convert a single day's row dict into a human-readable text chunk.
    This is what gets embedded — make it information-dense.
    """
    parts = []

    # Date and overall rating
    parts.append(f"Date: {row['date']}")
    parts.append(f"Rating: {row.get('ratingNum', 0)}/5")

    # Emotional state
    mood = row.get('mood', '')
    energy = row.get('energyLevel', '')
    anxiety = row.get('anxiety', '')
    stress = row.get('stress', '')
    reg = row.get('regulationQuality', '')
    if any([mood, energy, anxiety, stress, reg]):
        parts.append(f"Emotional state: mood={mood}, energy={energy}, anxiety={anxiety}, stress={stress}, regulation={reg}")

    if row.get('depressionFlag'):
        parts.append("Mood check-in flagged")

    brain_fog = row.get('brainFog', '')
    if brain_fog and brain_fog != 'None':
        parts.append(f"Brain fog: {brain_fog}")

    # Sleep
    sleep_time = row.get('sleepTime', 0)
    sleep_quality = row.get('sleepQuality', '')
    sleep_score = row.get('sleepScore', 0)
    if sleep_time > 0:
        parts.append(f"Sleep: {sleep_time}hrs, quality={sleep_quality}, score={sleep_score}/100")

    sleep_factors = row.get('sleepFactors', '')
    if sleep_factors:
        parts.append(f"Sleep factors: {sleep_factors}")

    bedtime = row.get('bedtimeRange', '')
    if bedtime:
        parts.append(f"Bedtime: {bedtime}")

    # Wearables
    hrv = row.get('overnightHRV', 0)
    bb_eod = row.get('bodyBattery', 0)
    bb_bod = row.get('bodyBatteryBOD', 0)
    rhr = row.get('rhr', 0)
    stress_score = row.get('garminStress', 0)
    wearable_parts = []
    if hrv > 0:       wearable_parts.append(f"HRV={hrv}ms")
    if bb_eod > 0:    wearable_parts.append(f"body_battery_EOD={bb_eod}")
    if bb_bod > 0:    wearable_parts.append(f"body_battery_BOD={bb_bod}")
    if rhr > 0:       wearable_parts.append(f"RHR={rhr}bpm")
    if stress_score > 0: wearable_parts.append(f"garmin_stress={stress_score}")
    if wearable_parts:
        parts.append(f"Wearables: {', '.join(wearable_parts)}")

    # Movement
    steps = row.get('steps', 0)
    activity = row.get('activityLevel', '')
    exercise = row.get('exercise', '')
    if steps > 0:
        parts.append(f"Steps: {steps:,}")
    if activity or exercise:
        parts.append(f"Movement: activity={activity}, exercise={exercise}")
    if row.get('sunlight'):
        parts.append("Got sunlight today")

    # Health
    symptoms = row.get('physicalSymptoms', '')
    tension = row.get('tensionCheck', '')
    pain = row.get('painLevel', '')
    hash_symp = row.get('hashimotosSymptoms', '')
    if symptoms and symptoms != 'None':
        parts.append(f"Physical symptoms: {symptoms}")
    if tension and tension != 'None':
        parts.append(f"Body tension: {tension}")
    if pain and pain != 'None':
        parts.append(f"Pain level: {pain}")
    if hash_symp and hash_symp not in ('', 'None'):
        parts.append(f"Hashimoto's symptoms: {hash_symp}")

    # Habits
    habits_on = []
    habit_map = {
        'meditation': 'meditation', 'journal': 'journaling', 'reading': 'reading',
        'gratitude': 'gratitude', 'yoga': 'yoga', 'kneeExercises': 'knee PT',
        'tarotPull': 'tarot',
    }
    for key, label in habit_map.items():
        if row.get(key):
            habits_on.append(label)
    if habits_on:
        parts.append(f"Habits done: {', '.join(habits_on)}")

    # Lifestyle
    diet = row.get('dietQuality', '')
    water = row.get('water', 0)
    caffeine = row.get('caffeine', '')
    if diet and diet != 'Neutral':
        parts.append(f"Diet: {diet}")
    if water > 0:
        parts.append(f"Water: {water}oz")
    if caffeine and caffeine != 'None':
        parts.append(f"Caffeine: {caffeine}")
    if row.get('alcohol'):
        parts.append("Had alcohol")
    if row.get('fastFood'):
        parts.append("Had fast food")
    if row.get('lateMeals'):
        parts.append("Ate late meals")

    # Social + environment
    social = row.get('socialInteractions', '')
    if social and social != 'Neutral':
        parts.append(f"Social interactions: {social}")
    weather = row.get('weather', '')
    if weather:
        parts.append(f"Weather: {weather}")

    # Mental patterns
    thinking = row.get('unhelpfulThinking', '')
    if thinking:
        parts.append(f"Cognitive patterns logged: {thinking}")

    # Productivity + finances
    productivity = row.get('productivity', '')
    financial = row.get('financial', '')
    if productivity:
        parts.append(f"Productivity: {productivity}")
    if financial:
        parts.append(f"Financial: {financial}")

    # Cycle
    if row.get('menstruating') and row.get('dayOfPeriod', 0) > 0:
        parts.append(f"Cycle tracking: day {row['dayOfPeriod']}")

    # Life events overlay
    if row.get('hasNegativeEvent'):
        parts.append("Active negative life event on this day")
    if row.get('hasPositiveEvent'):
        parts.append("Active positive life event on this day")
    if row.get('activeEventCount', 0) > 1:
        parts.append(f"{row['activeEventCount']} active life events overlapping")

    return '\n'.join(parts)
