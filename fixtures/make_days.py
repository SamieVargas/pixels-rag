"""Deterministic synthetic Life in Pixels rows.

The real export is personal data and stays off the repo. This fixture has the
same shape (the raw row fields chunk.py reads), a seeded generator, and a few
planted days the golden questions are written against. Regenerate with

    python fixtures/make_days.py

and the output is byte-identical every time.
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 7
START = date(2026, 5, 1)
END = date(2026, 8, 31)
UNLOGGED = {"2026-05-09", "2026-06-02", "2026-06-03", "2026-07-19", "2026-08-23"}

MOODS = ["Calm", "Content", "Flat", "Irritable", "Anxious", "Happy", "Low"]
LEVELS = ["Low", "Moderate", "High"]
SEVERITY = ["None", "Mild", "Moderate", "Severe"]
QUALITY = ["Poor", "Fair", "Good", "Excellent"]
WEATHER = ["Sunny", "Cloudy", "Rain", "Hot", "Storm"]
EXERCISE = ["", "Walking", "Hot yoga", "Hot yoga, Walking", "Strength training", "Cycling"]
CAFFEINE = ["None", "1 cup", "2 cups", "3+ cups"]
SOCIAL = ["Neutral", "Positive", "Draining", "Very positive"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def make_rows():
    rng = random.Random(SEED)
    rows = []
    d = START
    while d <= END:
        iso = d.isoformat()
        d += timedelta(days=1)
        if iso in UNLOGGED:
            continue
        weekend = date.fromisoformat(iso).weekday() >= 5
        exercise = rng.choices(EXERCISE, weights=[22, 26, 16, 14, 14, 8])[0]
        hot_yoga = "Hot yoga" in exercise
        walking = "Walking" in exercise
        alcohol = rng.random() < (0.22 if weekend else 0.08)
        late = rng.random() < 0.18
        meditation = rng.random() < 0.45
        # Sleep: hot yoga plus walking is the planted recovery pattern.
        sleep = 62 + rng.gauss(0, 9)
        if hot_yoga and walking:
            sleep += 13
        elif hot_yoga or walking:
            sleep += 5
        if alcohol:
            sleep -= 9
        if late:
            sleep -= 4
        sleep = int(clamp(round(sleep), 35, 96))
        bb = 45 + rng.gauss(0, 12) + (0.6 * (sleep - 62))
        if hot_yoga and walking:
            bb += 8
        bb = int(clamp(round(bb), 8, 98))
        rating = clamp(round(2.9 + (sleep - 62) / 18 + rng.gauss(0, 0.7) + (0.4 if weekend else 0)), 1, 5)
        steps = int(clamp(rng.gauss(7800, 2600) + (3500 if walking else 0), 1200, 18500))
        row = {
            "date": iso,
            "ratingNum": rating,
            "mood": rng.choice(MOODS) if rating < 4 else rng.choice(["Calm", "Content", "Happy"]),
            "energyLevel": LEVELS[clamp(int((sleep - 40) // 20), 0, 2)],
            "anxiety": rng.choices(SEVERITY, weights=[35, 35, 22, 8])[0],
            "stress": rng.choice(LEVELS),
            "regulationQuality": rng.choice(QUALITY),
            "depressionFlag": rating <= 1 and rng.random() < 0.5,
            "brainFog": rng.choices(SEVERITY, weights=[55, 25, 15, 5])[0],
            "sleepTime": round(clamp(5.2 + (sleep - 50) / 14 + rng.gauss(0, 0.5), 4.0, 9.5), 1),
            "sleepQuality": QUALITY[clamp(int((sleep - 40) // 15), 0, 3)],
            "sleepScore": sleep,
            "sleepFactors": "Late screen time" if late else "",
            "bedtimeRange": rng.choice(["10-11pm", "11pm-12am", "12-1am"]),
            "overnightHRV": int(clamp(rng.gauss(48, 10) + (sleep - 62) / 4, 20, 90)),
            "bodyBattery": bb,
            "bodyBatteryBOD": int(clamp(bb + rng.gauss(18, 8), 10, 100)),
            "rhr": int(clamp(rng.gauss(58, 4) - (2 if hot_yoga else 0), 46, 76)),
            "garminStress": int(clamp(rng.gauss(32, 10) - (sleep - 62) / 4, 8, 80)),
            "steps": steps,
            "activityLevel": "High" if steps > 11000 else ("Moderate" if steps > 6000 else "Low"),
            "exercise": exercise,
            "sunlight": rng.random() < 0.55,
            "physicalSymptoms": rng.choices(["None", "Headache", "Fatigue", "Sore"], weights=[65, 12, 13, 10])[0],
            "tensionCheck": rng.choices(["None", "Neck", "Shoulders", "Jaw"], weights=[50, 20, 20, 10])[0],
            "painLevel": rng.choices(["None", "Mild", "Moderate"], weights=[70, 22, 8])[0],
            "hashimotosSymptoms": "",
            "meditation": meditation,
            "journal": rng.random() < 0.35,
            "reading": rng.random() < 0.4,
            "gratitude": rng.random() < 0.3,
            "yoga": hot_yoga,
            "kneeExercises": rng.random() < 0.25,
            "tarotPull": rng.random() < 0.15,
            "dietQuality": rng.choices(["Neutral", "Good", "Poor"], weights=[50, 30, 20])[0],
            "water": int(clamp(rng.gauss(64, 18), 16, 120)),
            "caffeine": rng.choices(CAFFEINE, weights=[15, 45, 30, 10])[0],
            "alcohol": alcohol,
            "fastFood": rng.random() < 0.15,
            "lateMeals": late,
            "socialInteractions": rng.choices(SOCIAL, weights=[45, 30, 15, 10])[0],
            "weather": rng.choice(WEATHER),
            "unhelpfulThinking": rng.choices(["", "Catastrophizing", "All-or-nothing"], weights=[70, 18, 12])[0],
            "productivity": rng.choice(["Low", "Moderate", "High"]),
            "financial": rng.choices(["", "Spent more than planned"], weights=[85, 15])[0],
            "menstruating": False,
            "dayOfPeriod": 0,
            "hasNegativeEvent": False,
            "hasPositiveEvent": False,
            "activeEventCount": 0,
        }
        rows.append(row)

    by = {r["date"]: r for r in rows}

    # July holds exactly one five-star day, the positive-event day.
    for r in rows:
        if r["date"].startswith("2026-07") and r["ratingNum"] == 5:
            r["ratingNum"] = 4
    # Only one trail run in August, and a strong recovery the morning after.
    for r in rows:
        if "Trail" in r["exercise"]:
            r["exercise"] = "Cycling"

    plant = {
        "2026-06-14": {"physicalSymptoms": "Migraine", "ratingNum": 1, "sleepScore": 48, "sleepQuality": "Poor",
                       "brainFog": "Severe", "energyLevel": "Low", "mood": "Low", "exercise": "", "steps": 2100,
                       "activityLevel": "Low", "bodyBattery": 14},
        "2026-07-04": {"ratingNum": 5, "hasPositiveEvent": True, "activeEventCount": 1, "mood": "Happy",
                       "socialInteractions": "Very positive", "sunlight": True, "exercise": "Walking"},
        "2026-08-09": {"exercise": "Trail running", "steps": 16400, "activityLevel": "High"},
        "2026-08-10": {"sleepScore": 88, "sleepQuality": "Excellent", "bodyBattery": 91, "overnightHRV": 71,
                       "ratingNum": 5, "mood": "Content", "energyLevel": "High"},
        "2026-05-20": {"alcohol": True, "lateMeals": True, "sleepScore": 52, "sleepQuality": "Poor",
                       "sleepFactors": "Late screen time", "bodyBattery": 27, "ratingNum": 2},
        "2026-05-27": {"anxiety": "Severe", "stress": "High", "mood": "Anxious", "unhelpfulThinking": "Catastrophizing",
                       "ratingNum": 2, "hasNegativeEvent": True, "activeEventCount": 1},
        "2026-06-22": {"hashimotosSymptoms": "Fatigue, joint pain", "physicalSymptoms": "Fatigue", "energyLevel": "Low",
                       "ratingNum": 2},
        "2026-08-15": {"hashimotosSymptoms": "Fatigue, joint pain", "physicalSymptoms": "Fatigue", "energyLevel": "Low",
                       "ratingNum": 2},
    }
    for iso, patch in plant.items():
        by[iso].update(patch)
    # Only 2026-05-27 carries severe anxiety in May, and only 2026-05-20 pairs
    # alcohol with a late meal in May, so those questions have one right day.
    for r in rows:
        if r["date"].startswith("2026-05") and r["date"] != "2026-05-27" and r["anxiety"] == "Severe":
            r["anxiety"] = "Moderate"
        if r["date"].startswith("2026-05") and r["date"] != "2026-05-20" and r["alcohol"] and r["lateMeals"]:
            r["lateMeals"] = False
        if r["date"].startswith("2026-06") and r["date"] != "2026-06-14" and r["physicalSymptoms"] == "Migraine":
            r["physicalSymptoms"] = "Headache"
    return rows


def main():
    rows = make_rows()
    out = Path(__file__).with_name("days.json")
    out.write_text(json.dumps({"rows": rows, "totalRows": (END - START).days + 1}, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
