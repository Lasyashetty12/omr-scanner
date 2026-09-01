import os
import json
import urllib.request
import urllib.parse
from datetime import datetime


def _load_local_env():
    """Load env variables from local 'env' or '.env' file if not present in os.environ."""
    env_files = [
        os.path.join(os.path.dirname(__file__), "env"),
        os.path.join(os.path.dirname(__file__), ".env")
    ]
    for env_path in env_files:
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'").strip('"')
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass


_load_local_env()


def get_supabase_config():
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or ""
    return url, key


def is_db_configured():
    url, key = get_supabase_config()
    return bool(url and key)


def _supabase_request(endpoint, method="GET", data=None, query_params=None):
    url, key = get_supabase_config()
    if not url or not key:
        raise ValueError("Supabase database environment variables are missing.")

    full_url = f"{url}/rest/v1/{endpoint}"
    if query_params:
        full_url += "?" + urllib.parse.urlencode(query_params)

    req = urllib.request.Request(full_url, method=method)
    req.add_header("apikey", key)
    req.add_header("Authorization", f"Bearer {key}")
    req.add_header("Content-Type", "application/json")
    if method in ("POST", "PATCH", "PUT"):
        req.add_header("Prefer", "return=representation")

    body_bytes = None
    if data is not None:
        body_bytes = json.dumps(data).encode("utf-8")

    with urllib.request.urlopen(req, data=body_bytes) as response:
        resp_text = response.read().decode("utf-8")
        if resp_text:
            return json.loads(resp_text)
        return None


def save_omr_result_to_db(result_data, student_info=None):
    """
    Saves an OMR evaluation result to Supabase database tables:
    students, exams, omr_results, scans, question_results.
    Returns database result ID (int) or None if saving failed.
    """
    if not is_db_configured():
        return None

    try:
        # Default student info if not provided. Prefer identity bubbles read
        # from the OMR itself before falling back to generated placeholders.
        if not student_info:
            student_info = {}

        identity_data = result_data.get("identity") or {}

        student_payload = {
            "name": student_info.get("name") or "Student Candidate",
            "roll_number": (
                student_info.get("roll_number")
                or identity_data.get("roll_number")
                or result_data.get("roll_number")
                or f"ROLL-{result_data.get('scan_id', '000')[:6]}"
            ),
            "class_name": str(
                student_info.get("class_name")
                or identity_data.get("class")
                or result_data.get("class")
                or "12"
            ),
            "section": str(student_info.get("section") or "A"),
            "batch": str(student_info.get("batch") or "2026")
        }
        students_resp = _supabase_request("students", method="POST", data=student_payload)
        student_id = students_resp[0]["id"] if students_resp else None

        # Exam info
        exam_type = (result_data.get("exam") or "NEET").upper()
        paper_code = str(result_data.get("paper_code") or result_data.get("series") or "A1")
        exam_payload = {
            "exam_type": exam_type,
            "paper_code": paper_code,
            "paper_series": paper_code,
            "session": "Morning",
            "exam_date": datetime.utcnow().strftime("%Y-%m-%d")
        }
        exams_resp = _supabase_request("exams", method="POST", data=exam_payload)
        exam_id = exams_resp[0]["id"] if exams_resp else None

        # OMR Result Summary
        q_results = result_data.get("question_results") or {}
        total_questions = len(q_results) if q_results else 180

        omr_payload = {
            "student_id": student_id,
            "exam_id": exam_id,
            "score": float(result_data.get("score") or 0.0),
            "correct": int(result_data.get("correct") or 0),
            "wrong": int(result_data.get("wrong") or 0),
            "blank": int(result_data.get("blank") or 0),
            "multiple": int(result_data.get("multiple") or 0),
            "uncertain": int(result_data.get("uncertain") or 0),
            "total_questions": total_questions,
            "stream": str(result_data.get("stream") or "PCMB").upper(),
            "raw_result_json": json.dumps(result_data)
        }
        omr_resp = _supabase_request("omr_results", method="POST", data=omr_payload)
        if not omr_resp:
            return None
        omr_result_id = omr_resp[0]["id"]

        # Scan metadata
        scan_payload = {
            "omr_result_id": omr_result_id,
            "image_reference": result_data.get("scan_id") or "",
            "capture_source": "camera" if "camera" in str(result_data.get("original_image_url", "")) else "upload"
        }
        _supabase_request("scans", method="POST", data=scan_payload)

        # Question-wise breakdown
        if q_results:
            q_payloads = []
            for q_num_str, q_info in q_results.items():
                try:
                    q_num = int(q_num_str)
                except ValueError:
                    q_num = 0
                q_payloads.append({
                    "omr_result_id": omr_result_id,
                    "question_number": q_num,
                    "marked_answer": str(q_info.get("detected") or q_info.get("student_answer") or "—"),
                    "correct_answer": str(q_info.get("correct_answer") or q_info.get("answer") or "—"),
                    "status": str(q_info.get("status") or "Uncertain").capitalize()
                })
            if q_payloads:
                # Insert in chunks of 100 if necessary
                chunk_size = 100
                for i in range(0, len(q_payloads), chunk_size):
                    _supabase_request("question_results", method="POST", data=q_payloads[i:i+chunk_size])

        return omr_result_id

    except Exception as e:
        print("Database save error:", e)
        return None


def get_omr_results_from_db(class_filter=None, section_filter=None, exam_filter=None):
    """
    Retrieves all evaluated student OMR results from database with student & exam details.
    """
    if not is_db_configured():
        return []

    try:
        # Fetch omr_results select=*
        omr_records = _supabase_request("omr_results?select=*&order=created_at.desc", method="GET") or []
        if not omr_records:
            return []

        # Collect student_ids and exam_ids
        student_ids = list(set([r["student_id"] for r in omr_records if r.get("student_id")]))
        exam_ids = list(set([r["exam_id"] for r in omr_records if r.get("exam_id")]))

        students_map = {}
        if student_ids:
            # Query students
            s_query = f"students?id=in.({','.join(map(str, student_ids))})"
            students_list = _supabase_request(s_query, method="GET") or []
            students_map = {s["id"]: s for s in students_list}

        exams_map = {}
        if exam_ids:
            e_query = f"exams?id=in.({','.join(map(str, exam_ids))})"
            exams_list = _supabase_request(e_query, method="GET") or []
            exams_map = {e["id"]: e for e in exams_list}

        results = []
        for r in omr_records:
            s_data = students_map.get(r.get("student_id")) or {}
            e_data = exams_map.get(r.get("exam_id")) or {}

            student_name = s_data.get("name") or "Student Candidate"
            roll_number = s_data.get("roll_number") or f"ROLL-{r['id']}"
            class_name = s_data.get("class_name") or "12"
            section = s_data.get("section") or "A"
            exam_type = (e_data.get("exam_type") or "NEET").upper()

            # Apply filters if provided
            if class_filter and class_filter.lower() != "all" and class_name.lower() != class_filter.lower():
                continue
            if section_filter and section_filter.lower() != "all" and section.lower() != section_filter.lower():
                continue
            if exam_filter and exam_filter.lower() != "all" and exam_type.lower() != exam_filter.lower():
                continue

            paper_code = e_data.get("paper_code") or e_data.get("paper_series") or "A1"
            created_at = r.get("created_at") or datetime.utcnow().isoformat()

            # Keep the scan UUID available to the frontend.  It is the stable
            # public result key and can be resolved through the scans table even
            # when serverless local files are not on the same Vercel instance.
            scan_id = None
            if r.get("raw_result_json"):
                try:
                    raw_data = json.loads(r["raw_result_json"])
                    scan_id = raw_data.get("scan_id")
                except Exception:
                    scan_id = None

            results.append({
                "id": r["id"],
                "scan_id": scan_id,
                "student_name": student_name,
                "roll_number": roll_number,
                "class": class_name,
                "section": section,
                "batch": s_data.get("batch") or "2026",
                "exam": exam_type,
                "paper_code": paper_code,
                "score": r.get("score", 0),
                "correct": r.get("correct", 0),
                "wrong": r.get("wrong", 0),
                "blank": r.get("blank", 0),
                "multiple": r.get("multiple", 0),
                "uncertain": r.get("uncertain", 0),
                "total_questions": r.get("total_questions", 180),
                "stream": r.get("stream", "PCMB"),
                "date": created_at
            })

        return results

    except Exception as e:
        print("Database list error:", e)
        return []


def get_omr_result_by_scan_id_from_db(scan_id):
    """Resolve a scan UUID through scans.image_reference and return its OMR result."""
    if not is_db_configured() or not scan_id:
        return None

    try:
        encoded_scan_id = urllib.parse.quote(str(scan_id), safe="")
        scan_records = _supabase_request(
            f"scans?image_reference=eq.{encoded_scan_id}&select=omr_result_id&limit=1",
            method="GET",
        ) or []
        if not scan_records:
            return None

        omr_result_id = scan_records[0].get("omr_result_id")
        if omr_result_id is None:
            return None

        return get_omr_result_by_id_from_db(omr_result_id)
    except Exception as e:
        print("Database get by scan id error:", e)
        return None


def get_omr_result_by_id_from_db(result_id):
    """
    Fetches detailed result for a single result_id from database.
    """
    if not is_db_configured():
        return None

    try:
        omr_records = _supabase_request(f"omr_results?id=eq.{result_id}", method="GET") or []
        if not omr_records:
            return None

        r = omr_records[0]

        # Fetch student and exam
        s_data = {}
        if r.get("student_id"):
            s_list = _supabase_request(f"students?id=eq.{r['student_id']}", method="GET") or []
            if s_list:
                s_data = s_list[0]

        e_data = {}
        if r.get("exam_id"):
            e_list = _supabase_request(f"exams?id=eq.{r['exam_id']}", method="GET") or []
            if e_list:
                e_data = e_list[0]

        # Fetch question results
        q_list = _supabase_request(f"question_results?omr_result_id=eq.{result_id}&order=question_number.asc", method="GET") or []

        q_dict = {}
        for q in q_list:
            q_num = str(q.get("question_number", 0))
            q_dict[q_num] = {
                "question_number": q.get("question_number"),
                "detected": q.get("marked_answer"),
                "student_answer": q.get("marked_answer"),
                "correct_answer": q.get("correct_answer"),
                "status": q.get("status")
            }

        # Parse raw JSON if available for extra image references or scores
        raw_data = {}
        if r.get("raw_result_json"):
            try:
                raw_data = json.loads(r["raw_result_json"])
            except Exception:
                pass

        scan_id = raw_data.get("scan_id") or f"id_{r['id']}"

        result_obj = {
            "id": r["id"],
            "scan_id": scan_id,
            "status": "processed",
            "exam": (e_data.get("exam_type") or "NEET").upper(),
            "stream": r.get("stream", "PCMB"),
            "paper_code": e_data.get("paper_code") or e_data.get("paper_series") or "A1",
            "score": r.get("score"),
            "correct": r.get("correct"),
            "wrong": r.get("wrong"),
            "blank": r.get("blank"),
            "multiple": r.get("multiple"),
            "uncertain": r.get("uncertain"),
            "total_questions": r.get("total_questions"),
            "student": {
                "name": s_data.get("name") or "Student Candidate",
                "roll_number": s_data.get("roll_number") or f"ROLL-{r['id']}",
                "class": s_data.get("class_name") or "12",
                "section": s_data.get("section") or "A",
                "batch": s_data.get("batch") or "2026"
            },
            "exam_info": {
                "exam_type": (e_data.get("exam_type") or "NEET").upper(),
                "paper_code": e_data.get("paper_code") or "A1",
                "paper_series": e_data.get("paper_series") or "A1",
                "exam_date": e_data.get("exam_date") or r.get("created_at", "")[:10],
                "session": e_data.get("session") or "Morning"
            },
            "question_results": q_dict if q_dict else raw_data.get("question_results", {}),
            "original_image_url": raw_data.get("original_image_url") or f"/uploads/{scan_id}.jpg",
            "corrected_image_url": raw_data.get("corrected_image_url") or f"/uploads/{scan_id}.jpg",
            # Debug images are optional and may not exist (especially on
            # ephemeral/serverless storage). Never invent a URL for a file
            # that was not actually recorded with the result.
            "bubble_debug_image_url": raw_data.get("bubble_debug_image_url")
        }

        return result_obj

    except Exception as e:
        print("Database get by id error:", e)
        return None
