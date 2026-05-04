"""
Adversarial accuracy test for the fake job detector.

20 real + 20 fake job postings, none matching the 5 synthetic templates
in extract_dataset.py. Tests how the model generalizes beyond memorized
training distribution.

Run with the API already started, OR directly:
    cd backend && ../.venv/bin/python ../tests/adversarial_test.py
"""

import sys
import os

# Make backend importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import numpy as np  # noqa: E402
from predict import predict, fused_score  # noqa: E402


# =====================================================
# REAL JOB POSTINGS (label = 0)
# =====================================================
# Diverse: tech, healthcare, trades, retail, government, education,
# different lengths, formal and informal styles.
REAL_JOBS = [
    # 1 — Long structured tech
    """Senior Data Engineer
    Stripe is hiring a Senior Data Engineer to build and scale our payments
    data platform. You will own pipelines that process billions of events per
    day, design schemas for a multi-petabyte warehouse, and partner with
    finance, fraud, and product teams.
    Responsibilities: design and operate batch and streaming pipelines; own
    data quality; mentor junior engineers.
    Requirements: 6+ years of data engineering experience; expertise in
    Spark, Airflow, dbt; strong SQL; experience with AWS or GCP.
    Stripe offers competitive compensation, equity, and comprehensive
    healthcare. Apply via stripe.com/jobs.""",

    # 2 — Healthcare, formal
    """Registered Nurse — Pediatric Oncology
    Boston Children's Hospital is recruiting a Registered Nurse for our
    Pediatric Oncology floor. Active MA RN license required. BSN preferred.
    Two years of acute care experience required; pediatric oncology
    experience strongly preferred. Shifts: 7p-7a, three nights per week.
    Magnet-designated facility. Benefits include tuition reimbursement and
    pension. Apply at childrenshospital.org/careers.""",

    # 3 — Trades / blue collar
    """HVAC Service Technician
    Atlas Mechanical is hiring an experienced HVAC technician for
    residential service calls in the Phoenix area. EPA Universal certification
    required. Must have own hand tools and a valid Arizona driver's license.
    Company truck, fuel card, uniforms, and tablet provided. Pay $28-35/hr
    DOE plus quarterly bonus. Send resume and certifications to
    careers@atlasmechanical-az.com.""",

    # 4 — Retail
    """Store Manager — Trader Joe's, Brooklyn
    Lead a 60-person crew at our Park Slope store. Five years of retail
    management experience, ideally in grocery or specialty food. Strong
    people leadership and inventory planning skills. Annual base $78,000-
    $92,000 plus benefits. Internal candidates encouraged to apply.""",

    # 5 — Government / civil service
    """City Planner II — Department of Urban Planning
    The City of Austin is accepting applications for a City Planner II in
    the Comprehensive Planning division. Master's degree in urban planning
    or related field required. Three years of professional planning
    experience. AICP certification preferred. Salary range $68,432-$87,360.
    Apply through austintexas.gov/jobs by June 14.""",

    # 6 — Teaching
    """Adjunct Professor of Computer Science (Spring 2027)
    Northeastern University seeks adjunct faculty to teach CS2510
    (Fundamentals of Computer Science 2). PhD or near-completion preferred;
    Master's with industry experience considered. One section, three credit
    hours. Apply via northeastern.edu/employment.""",

    # 7 — Short, informal but legit
    """Hiring barista at Onyx Coffee, Fayetteville. Mornings, weekends.
    No experience needed, training provided. Free pound of coffee per week.
    Stop by the shop with a resume or apply online at onyxcoffeelab.com.""",

    # 8 — Internship
    """Software Engineering Intern — Summer 2027
    Cloudflare is recruiting interns for our Austin office. Must be enrolled
    in a Bachelor's or Master's program in CS or related. Familiarity with
    one systems language (Rust, Go, C++). Twelve-week paid internship.
    Apply at cloudflare.com/careers/early-talent.""",

    # 9 — Sales
    """Account Executive, Mid-Market
    HubSpot is hiring an AE for the New England territory. 3-5 years SaaS
    sales experience, CRM or marketing tech preferred. OTE $145k-$175k.
    Hybrid out of Cambridge HQ.""",

    # 10 — Long technical
    """Principal Site Reliability Engineer
    Datadog is looking for a Principal SRE to drive reliability for our
    metrics ingestion pipeline. You will own SLOs, lead post-incident
    reviews, and design capacity for tens of millions of host-seconds per
    minute.
    Requirements: 10+ years of production engineering, deep experience with
    Linux, container orchestration, distributed storage. Track record of
    leading reliability programs at scale. Compensation: $260k-$340k base
    plus RSU. Remote-friendly within US time zones.""",

    # 11 — Construction
    """Project Superintendent — Commercial Construction
    Skanska USA is seeking a Superintendent for a $40M healthcare
    renovation in Seattle. OSHA 30, ten years of commercial superintendent
    experience. Travel to job site daily. Union project. Apply at
    skanska.com/usa/careers.""",

    # 12 — NGO
    """Program Officer — Global Health Equity
    The Bill & Melinda Gates Foundation is hiring a Program Officer in our
    Seattle office. Master's degree in public health or related; 7+ years
    in international development or global health. Up to 30% international
    travel. Salary range $145,000-$210,000.""",

    # 13 — Lab technician
    """Research Associate II — Cell Biology
    Genentech is recruiting a Research Associate for our Oncology
    Discovery group in South San Francisco. BS or MS in cell biology,
    biochemistry, or related. Two years of mammalian cell culture
    experience. Apply at gene.com/careers.""",

    # 14 — Customer service
    """Customer Support Specialist — Tier 2
    Zapier is hiring a Tier 2 Support Specialist. Fully remote within North
    America. Two years of SaaS support experience required. Excellent
    written communication. Salary $62,000-$78,000 USD.""",

    # 15 — Logistics
    """Warehouse Associate — Night Shift
    Costco Wholesale Distribution Center, Mira Loma CA. 9pm-5:30am, four
    nights per week. Must be able to lift 50 lbs repeatedly. $24.85/hr
    starting plus shift differential. Full benefits after 90 days.""",

    # 16 — Editor
    """Senior Copy Editor
    The Atlantic is hiring a Senior Copy Editor for our digital newsroom.
    Five years of editing experience at a major publication. Deep
    understanding of AP and house style. Hybrid out of Washington DC.
    Apply with cover letter and three samples to careers@theatlantic.com.""",

    # 17 — UX
    """Senior Product Designer
    Figma seeks a Senior Product Designer for our Auto-Layout team.
    Portfolio required; show systems thinking and craft. Six years of
    product design experience. SF or NYC, hybrid 3 days in office.""",

    # 18 — Finance
    """Senior Financial Analyst
    Vanguard is hiring a Senior Financial Analyst in Malvern, PA. Bachelor's
    in finance or accounting; CPA or CFA progression preferred. 4-6 years
    of FP&A experience. Apply at vanguardjobs.com.""",

    # 19 — Bilingual support
    """Bilingual Spanish-English Case Manager
    Catholic Charities of Chicago is recruiting a bilingual case manager
    for our refugee resettlement program. MSW preferred; BSW with three
    years experience considered. $52,000-$58,000 plus benefits.""",

    # 20 — Casual posting, real
    """Looking for a part-time line cook at Lula Cafe, Logan Square.
    Two years of restaurant experience. Wednesday-Sunday, 7am-3pm shifts.
    $22/hr plus pooled tips. Email chris@lulacafe.com or come in
    Tuesday-Friday between 2 and 4pm.""",
]


# =====================================================
# FAKE / SCAM JOB POSTINGS (label = 1)
# =====================================================
# Deliberately diverse: NONE match the 5 hardcoded templates in
# extract_dataset.py. We include scam types absent from training:
# crypto/MLM, reshipping, fake government, identity-collection,
# check-cashing, recruiter impersonation, plus newer keyword-light scams.
FAKE_JOBS = [
    # 1 — Reshipping / package mule
    """Quality Control Inspector — Work From Anywhere
    Receive packages at home, photograph contents, repackage and forward
    using prepaid labels we send you. Average 3 hours/day. Earn $2,800
    weekly. We pay shipping. Reply with full name, home address, and SSN
    for our background check to hr.qcinspector@protonmail.com.""",

    # 2 — Fake check / payroll scam
    """Personal Assistant Needed — Flexible Hours
    Busy executive seeking a virtual personal assistant. Tasks include
    purchasing gift cards for clients, depositing checks I send you, and
    sending the balance via Zelle to my preferred vendors. $750/week.
    Email davidstone.exec@aol.com to start immediately.""",

    # 3 — Crypto MLM disguised as job
    """Trader Position — Earn $5K-$15K Monthly
    Join our network of independent traders. No experience required, our
    AI does the work. Initial activation: $399 (refundable after first
    payout). Recruit two friends, get yours free. DM @cryptotrader_pro
    on Instagram for details.""",

    # 4 — Fake government scam
    """U.S. Postal Service — Mail Sorter Position
    USPS is hiring mail sorters at $26/hr, no experience needed. To
    secure your spot, pay the $89 testing fee at postalexamprep-2027.com
    and complete the practice exam today. Limited positions, apply now!""",

    # 5 — Identity-harvesting "interview"
    """Hiring Now: Remote Customer Service — Apple Authorized Reseller
    Position pays $35/hr from home. To begin onboarding please send a
    photo of your driver's license, social security card, and a voided
    check to verify direct deposit at apple-authorized-hr@gmail.com.""",

    # 6 — Pyramid scheme
    """Brand Ambassador (No Cap on Earnings)
    Promote luxury wellness products on social media. Earn 30% on personal
    sales and 10% on your downline. Starter kit $499 (lifetime access).
    Top earners make six figures part-time. Tap link in bio.""",

    # 7 — Subtle scam, no SCAM_WORDS hits
    """Administrative Coordinator — Fortune 500 Subsidiary
    Our parent company is expanding U.S. operations and needs an
    Administrative Coordinator immediately. We will Fedex your equipment
    and a check for office setup before you start; deposit it and wire
    $1,800 to our IT vendor for the laptop configuration. The remainder
    is your signing bonus. Reply with your address.""",

    # 8 — Fake recruiter / Indeed-style
    """Sarah from TalentBridge here! I came across your profile and we have
    an urgent opening for a Project Coordinator at a major bank, fully
    remote, $95k. Skip the interview — they trust our screening. Just
    pay $250 for our background check service and you're in. Cashapp
    $sarah_talentbridge.""",

    # 9 — Cam/escort under "modeling"
    """Female Models Needed — Easy Money
    Premium online platform recruiting attractive females 18-35. Work from
    your bedroom, set your own hours, $3K-$10K weekly possible. Apply with
    full body photos and bank info to onlyfans-recruiters@yandex.com.""",

    # 10 — Romance-pivot job
    """Companion / Personal Assistant for Wealthy Gentleman
    Mature professional seeks a discreet companion to handle scheduling
    and travel arrangements. $4,000/week, all expenses paid. Send a
    recent photo and short bio to henry.morgan.ceo@mail.ru.""",

    # 11 — Resume harvesting
    """Multiple Positions Open — Major Tech Company (NDA Required)
    We are conducting confidential hiring for a major Silicon Valley firm.
    Send your resume, two recent paystubs, and a copy of your passport to
    confidential.tech.recruiter@yahoo.com to be considered.""",

    # 12 — Investment "advisor"
    """Junior Investment Advisor — No License Required
    Help clients trade forex and crypto. We provide leads. Earn 60%
    commission. To unlock the trading platform, deposit $500 to your
    company-issued account. Most advisors clear $8K in their first month.""",

    # 13 — Subtle, professional looking
    """Bookkeeper — Small Engineering Firm (Remote, 1099)
    Looking for a part-time bookkeeper to reconcile accounts. Pay $40/hr.
    To get started, fill out our W-9 and provide your routing and account
    number along with online banking login so we can configure direct
    deposit ourselves.""",

    # 14 — Survey scam
    """Paid Product Tester — Earn $200 Per Test
    Test products like iPhones, AirPods, and PS5 and keep them after.
    A small $35 verification fee unlocks your tester account. Pay via
    PayPal Friends & Family to producttesterhq@gmail.com.""",

    # 15 — MLM essential oils
    """Wellness Consultant — Be Your Own Boss
    Join our team of 50,000 wellness consultants! No experience needed.
    Starter kit only $199. Earn commissions and free product. Build your
    team. Top consultants earn $20K/month. DM me on Facebook.""",

    # 16 — Fake job board
    """Human Resources Coordinator (Hiring 50 Positions)
    Major retail chain hiring nationwide. Click recruit-link.tk/apply
    to claim your interview slot. A $19.99 application processing fee
    is required to verify your seriousness about the position.""",

    # 17 — Reship vehicle title scam
    """Vehicle Wrapping — Get Paid to Drive
    Pepsi is paying drivers $600/week to wrap their personal vehicles in
    advertising. We will mail you a check for $4,800; deposit it,
    keep $600 as your first payment, and Zelle the remaining $4,200
    to our wrapping contractor. Apply with your home address.""",

    # 18 — Mystery shopper
    """Mystery Shopper Wanted — $400 Per Assignment
    Visit local Western Union branches and rate the customer service.
    We send you a check for $2,500. Cash it, wire $2,000 via Western
    Union to our QC department, keep the rest. No experience needed,
    must be 18+.""",

    # 19 — Fake job offer email style
    """Dear Candidate,
    Congratulations! Your resume has been shortlisted for the position
    of Logistics Manager at Amazon (annual package $115,000). To proceed
    with onboarding, please remit the refundable $250 documentation
    processing fee to amazon.hr.onboarding@outlook.com via PayPal.""",

    # 20 — Cover-fee scam without SCAM_WORDS
    """Commercial Driver — Long Haul (Class A)
    Hiring CDL-A drivers for Midwest routes. $0.78 CPM, sign-on bonus
    $7,500. Orientation in Joplin, MO — pay $145 for the orientation
    materials and DOT physical voucher (refunded after first dispatch)
    to secure your seat.""",
]


# =====================================================
# RUN
# =====================================================

def evaluate():
    print(f"\nEvaluating {len(REAL_JOBS)} real + {len(FAKE_JOBS)} fake = "
          f"{len(REAL_JOBS) + len(FAKE_JOBS)} postings\n")

    THRESHOLD_FAKE = 0.50       # api.py: fused score > 50 -> FAKE JOB
    THRESHOLD_REAL = 0.30       # api.py: fused score < 30 -> REAL JOB

    rows = []
    y_true = []
    y_pred_binary = []   # 0 = real, 1 = fake (using 0.5 cutoff on fused prob)
    y_pred_threetier = []  # 'REAL'/'SUSPICIOUS'/'FAKE' at api.py thresholds

    def score_one(text, true_label):
        bert_p, fused_p, ev = fused_score(text)
        y_true.append(0 if true_label == "REAL" else 1)
        y_pred_binary.append(1 if fused_p >= 0.5 else 0)
        if fused_p > THRESHOLD_FAKE:
            tier = "FAKE"
        elif fused_p < THRESHOLD_REAL:
            tier = "REAL"
        else:
            tier = "SUSPICIOUS"
        y_pred_threetier.append(tier)
        rows.append((true_label, bert_p, fused_p, ev, tier, text[:55].replace("\n", " ")))

    for text in REAL_JOBS:
        score_one(text, "REAL")
    for text in FAKE_JOBS:
        score_one(text, "FAKE")

    # ---- per-row report ----
    print(f"{'TRUE':<6} {'BERT':>6} {'FUSED':>6} {'EV':>3}  {'TIER':<11} TEXT")
    print("-" * 110)
    for true_label, bert_p, fused_p, ev, tier, text in rows:
        correct = (true_label == "FAKE" and fused_p >= 0.5) or \
                  (true_label == "REAL" and fused_p < 0.5)
        marker = "✓" if correct else "✗"
        print(f"{true_label:<6} {bert_p*100:>5.1f}% {fused_p*100:>5.1f}% {ev:>3}  {tier:<11} {marker} {text}")

    # ---- binary metrics at 0.5 cutoff ----
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred_binary)

    tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
    tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())

    accuracy = (tp + tn) / len(y_true_arr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print("\n" + "=" * 60)
    print("BINARY METRICS  (cutoff = 0.5)")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.3f}  ({tp + tn}/{len(y_true_arr)})")
    print(f"Precision: {precision:.3f}  (TP / (TP+FP))")
    print(f"Recall   : {recall:.3f}  (TP / (TP+FN))")
    print(f"F1       : {f1:.3f}")
    print()
    print("Confusion matrix:")
    print(f"                pred REAL  pred FAKE")
    print(f"  true REAL        {tn:>4}        {fp:>4}")
    print(f"  true FAKE        {fn:>4}        {tp:>4}")

    # ---- three-tier (api.py) view ----
    print("\n" + "=" * 60)
    print("THREE-TIER VIEW  (api.py thresholds: <35 REAL, >65 FAKE)")
    print("=" * 60)
    tiers = ["REAL", "SUSPICIOUS", "FAKE"]
    for label_name, label_int in [("REAL postings", 0), ("FAKE postings", 1)]:
        counts = {t: 0 for t in tiers}
        for true_l, tier in zip(y_true, y_pred_threetier):
            if true_l == label_int:
                counts[tier] += 1
        total = sum(counts.values())
        print(f"  {label_name:<16}", end="")
        for t in tiers:
            print(f"  {t}:{counts[t]:>2}/{total}", end="")
        print()


if __name__ == "__main__":
    evaluate()
