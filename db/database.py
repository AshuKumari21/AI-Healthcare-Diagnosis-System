
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "healthcare.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fullname TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    # Assessments Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        fullname TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT NOT NULL,
        reg_id TEXT NOT NULL,
        health_issue TEXT NOT NULL,
        submission_date TEXT NOT NULL,
        FOREIGN KEY (user_email) REFERENCES users (email)
    )
    ''')
    
    # Patient Registrations Table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patient_registrations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT NOT NULL,
        mobile TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        prediction_type TEXT NOT NULL,
        registration_date TEXT NOT NULL
    )
    ''')

    # Seed baseline patient cohort registrations if empty or low
    try:
        cursor.execute("SELECT COUNT(*) FROM patient_registrations")
        if cursor.fetchone()[0] < 4:
            demo_patients = [
                ("Eleanor Vance", "eleanor.v@care.org", "+1-555-0142", 58, "Female", "Stroke Risk Screening", "2026-03-01"),
                ("Marcus Sterling", "m.sterling@health.net", "+1-555-0188", 64, "Male", "Cardiovascular Track", "2026-03-02"),
                ("Ananya Patel", "ananya.p@clinic.in", "+1-555-0129", 45, "Female", "Type-2 Diabetes Screening", "2026-03-03"),
                ("David Kim", "dkim77@wellness.org", "+1-555-0174", 52, "Male", "Renal & Kidney Health", "2026-03-04"),
                ("Sophia Rodriguez", "sophia.r@medicare.com", "+1-555-0193", 39, "Female", "Hepatic / Liver Profile", "2026-03-05"),
                ("Robert Taylor", "rtaylor@globalhealth.org", "+1-555-0115", 71, "Male", "Stroke Risk Screening", "2026-03-06"),
                ("Aisha Al-Mansoor", "aisha.m@careplus.org", "+1-555-0166", 49, "Female", "Cardiovascular Track", "2026-03-07"),
                ("Carlos Mendez", "cmendez@mednet.org", "+1-555-0131", 62, "Male", "Hypertension & BP Track", "2026-03-08")
            ]
            cursor.executemany('''
            INSERT INTO patient_registrations (full_name, email, mobile, age, gender, prediction_type, registration_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', demo_patients)
    except Exception as e:
        print(f"Seed patient registration warning: {e}")

    conn.commit()
    conn.close()

def add_patient_registration(full_name, email, mobile, age, gender, prediction_type, reg_date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO patient_registrations (full_name, email, mobile, age, gender, prediction_type, registration_date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (full_name, email, mobile, age, gender, prediction_type, reg_date))
    conn.commit()
    conn.close()

def get_all_patient_registrations():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, email, mobile, age, gender, prediction_type, registration_date FROM patient_registrations ORDER BY id DESC")
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "name": r[1], "email": r[2], "mobile": r[3], "age": r[4], "gender": r[5], "prediction": r[6], "date": r[7]} for r in rows]
    except sqlite3.OperationalError:
        return []

def delete_patient_registration(patient_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM patient_registrations WHERE id = ?", (patient_id,))
    conn.commit()
    conn.close()

def add_user(fullname, phone, email, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (fullname, phone, email, password) VALUES (?, ?, ?, ?)",
                       (fullname, phone, email, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user(email, password):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, fullname, phone, email FROM users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user[0], "fullname": user[1], "phone": user[2], "email": user[3]}
    return None

def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, fullname, phone, email FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return {"id": user[0], "fullname": user[1], "phone": user[2], "email": user[3]}
    return None

def add_assessment(user_email, fullname, phone, email, reg_id, health_issue, date):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO assessments (user_email, fullname, phone, email, reg_id, health_issue, submission_date)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_email, fullname, phone, email, reg_id, health_issue, date))
    conn.commit()
    conn.close()

def get_assessment_history(user_email):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT fullname, phone, email, reg_id, health_issue, submission_date FROM assessments WHERE user_email = ? ORDER BY id DESC", (user_email,))
    history = cursor.fetchall()
    conn.close()
    return [{"name": h[0], "phone": h[1], "email": h[2], "reg_id": h[3], "issue": h[4], "date": h[5]} for h in history]

if __name__ == "__main__":
    init_db()
