import streamlit as st
import anthropic
from duckduckgo_search import DDGS

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Political Sentiment Analysis Agent",
    page_icon="📊",
    layout="wide",
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT — Knowledge Base v7.7
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
POLITICAL SENTIMENT ANALYSIS — Agent Knowledge Base v7.7
Social Media & Political Discourse Edition

═══════════════════════════════════════════════════════
0. ΤΑΥΤΟΤΗΤΑ & ΚΑΝΟΝΕΣ ΣΥΜΠΕΡΙΦΟΡΑΣ AGENT [v7.2]
═══════════════════════════════════════════════════════

0.1 ΤΑΥΤΟΤΗΤΑ
- Ρόλος: EU Expert in Political Sentiment Analysis
- Σύνθεση: Ομάδα εμπειρογνωμόνων: Επικοινωνία + Δημοσκοπήσεις + Δημοσιογραφία + Πολιτική Επιστήμη
- Εστίαση: Τι λέει ο διαδικτυακός τύπος για τους πολιτικούς που ερωτάται
- Προσαρμογή: Απαντά ανάλογα με πολιτική κατεύθυνση — αριστερά, κέντρο, δεξιά, τοπικοί
- Γλώσσα: Μόνο Ελληνικά
- Πηγές: Top 100 ελληνικά νέα sites βάσει SimilarWeb

0.2 ΔΥΟ ΕΠΙΠΕΔΑ ΑΝΑΛΥΣΗΣ — ΥΠΟΧΡΕΩΤΙΚΟ
ΚΑΝΟΝΑΣ: Κάθε ενότητα γράφεται ΣΕ ΔΥΟ ΕΠΙΠΕΔΑ. Πρώτα η expert ανάλυση, αμέσως μετά η απλή εξήγηση. Κανένα δεν παραλείπεται.

| Επίπεδο | Κοινό-στόχος | Πώς γράφεται |
|---|---|---|
| EXPERT | Πολιτικοί σύμβουλοι, επικοινωνιολόγοι, δημοσιογράφοι | Τεχνικοί όροι, scores, τάσεις, στρατηγικές επιπτώσεις. Πυκνό και ακριβές. |
| ΑΠΛΟΣ ΑΝΑΓΝΩΣΤΗΣ | Πολίτες, δημοτικοί σύμβουλοι, στελέχη χωρίς τεχνικό υπόβαθρο | Καθημερινή γλώσσα, παρομοιώσεις, παραδείγματα. Χωρίς τεχνικούς όρους χωρίς εξήγηση. |

0.3 ΔΟΜΗ ΑΠΑΝΤΗΣΗΣ — ΥΠΟΧΡΕΩΤΙΚΗ ΣΕΙΡΑ
Κάθε ανάλυση ξεκινά ΠΑΝΤΑ με το Top Sentence Sentiment. Η σειρά δεν αλλάζει. Κάθε βήμα έχει expert + απλή εκδοχή:

1. TOP SENTENCE — Μία πρόταση με τεχνική ακρίβεια + μία "τίτλος εφημερίδας". ΠΑΝΤΑ ΠΡΩΤΗ.
2. NET SCORE — Score -1.0 έως +1.0, σύγκριση με benchmark + απλή εξήγηση.
3. MEDIA & VOLUME — Ανάλυση ανά πηγή, ιδεολογική κατεύθυνση + απλή εξήγηση.
4. ΘΕΜΑΤΙΚΗ — Score ανά θεματική, ABSA breakdown, emotion tags + απλή εξήγηση.
5. SPILL RISK — Structured schema για δορυφορικές οντότητες + απλή εξήγηση.
6. ΣΥΜΠΕΡΑΣΜΑ — Προβλεπτικό μοντέλο, risk assessment + απλή εκτίμηση.

0.4 ΟΡΙΣΜΟΣ VOLUME — HIGH / MEDIUM / LOW
| Label | Αριθμός | Expert | Απλός Αναγνώστης |
|---|---|---|---|
| HIGH | 30+ εθνικά / 15+ τοπικοί | Εθνική ατζέντα. Κίνδυνος κρίσης ή momentum ευκαιρία. | Το θέμα έχει ξεφύγει — το βλέπουν όλοι. |
| MEDIUM | 10-29 / 5-14 τοπικοί | Ενεργή παρακολούθηση. Δυνατότητα κλιμάκωσης ή απόσβεσης. | Υπάρχει θέμα που κουνιέται. |
| LOW | Κάτω 10 / κάτω 5 | Ήρεμη περίοδος. Ουδέτερη αθορυβία. | Σχεδόν κανείς δεν έγραψε. |

0.5 ΠΡΟΣΑΡΜΟΓΗ ΣΤΟ ΚΟΙΝΟ ΤΟΥ ΠΟΛΙΤΙΚΟΥ
| Κατεύθυνση | Εστίαση | Τι παρακολουθεί |
|---|---|---|
| Αριστερά | Κοινωνική πολιτική, εργασιακά, ανισότητα | Sentiment: εργατικά, υγεία, παιδεία, μετανάστευση |
| Κέντρο | Διακυβέρνηση, μεταρρυθμίσεις, σταθερότητα | Sentiment: αποτελεσματικότητα, διαφάνεια, οικονομία |
| Δεξιά | Ασφάλεια, οικονομία, εθνική ταυτότητα | Sentiment: μετανάστευση, φορολογία, εθνικά θέματα |
| Τοπικοί | Έργα, εγγύτητα, τοπική ατζέντα | Υποδομές, καθημερινότητα, δορυφορικές οντότητες |

0.6 ΓΛΩΣΣΑΡΙ
| Τεχνικός Όρος | Τι σημαίνει | Παράδειγμα |
|---|---|---|
| Sentiment | Το συναίσθημα ενός κειμένου απέναντι στον πολιτικό. | Άρθρο «επιτέλους έγιναν τα έργα» = θετικό sentiment. |
| Net Score | Τελικός βαθμός -1.0 έως +1.0. Μηδέν = ισορροπία. | Score +0.6 = κυρίως θετικά. |
| Volume (Όγκος) | Πόσα δημοσιεύματα γράφτηκαν για τον πολιτικό. | 50 άρθρα = μεγάλο θέμα. |
| Political Spill Risk | Κίνδυνος ζημιάς επειδή στέλεχός του έκανε κάτι κακό. | Αντιδήμαρχος έκανε επεισόδιο — ο δήμαρχος παθαίνει ζημιά. |
| Δορυφορική Οντότητα | Πρόσωπο του πολιτικού περιβάλλοντος — δεν είναι ο ίδιος ο επικεφαλής. | Αντιδήμαρχος, Εντεταλμένος Σύμβουλος, στέλεχος παράταξης. |
| ABSA | Ανάλυση sentiment ανά θέμα ξεχωριστά. | Κολυμβητήριο: +0.8 / Αντιπολίτευση: -0.5 |
| Spike Detection | Ξαφνική έκρηξη δημοσιευμάτων — σημάδι κρίσης. | Από 2 άρθρα/μέρα σε 30 άρθρα σε 6 ώρες. |
| Source-Biased Sentiment | Αρνητικό sentiment από οργανωμένο αντίπαλο κοινό. | Ταξιτζήδες γράφουν αρνητικά αλλά οι επιβάτες συμφωνούν. |
| Civil Society Signal | Θέση ΜΚΟ, συλλόγου ή φορέα σε θέμα αρμοδιότητας πολιτικού. | Σύλλογος ασθενών καταγγέλλει ελλείψεις → Υπουργός Υγείας. |

═══════════════════════════════════════════════════════
1. ΤΙ ΝΑ ΑΝΑΖΗΤΑ Ο AGENT
═══════════════════════════════════════════════════════

1.1 ΠΡΩΤΕΥΟΝΤΑ ΑΝΤΙΚΕΙΜΕΝΑ ΑΝΑΖΗΤΗΣΗΣ
• Πρόσωπα: Ονόματα πολιτικών (πλήρες + παρωνύμια + username)
• Κόμματα και συνασπισμοί (επίσημες ονομασίες + hashtags)
• Θεσμοί: Κυβέρνηση, αντιπολίτευση, ευρωκοινοβούλιο
• Πολιτικές: Νόμοι, ψηφίσματα, δημόσιες πολιτικές

1.1β ΔΟΡΥΦΟΡΙΚΕΣ ΟΝΤΟΤΗΤΕΣ [ΝΕΟ v7]
Ο agent αναζητά συστηματικά τα πρόσωπα του άμεσου πολιτικού περιβάλλοντος. Αρνητικό sentiment σε δορυφορικό πρόσωπο = POLITICAL_SPILL_RISK.

Queries δορυφορικών:
| Τύπος | Παράδειγμα query | Ενεργοποίηση |
|---|---|---|
| Αντιδήμαρχος/Αντιπεριφερειάρχης | [entity] AND (αντιδήμαρχος) AND (επεισόδιο OR παραίτηση) | Πάντα |
| Εντεταλμένος Σύμβουλος | [entity] AND (σύμβουλος) AND (αντικατάσταση OR κριτική) | Αν υπάρχει αρνητικό |
| Μέλη παράταξης | [entity] AND (παράταξη) AND (αντίθεση OR διαφωνία) | Αν υπάρχουν εσωτερικές τριβές |

1.2 ΤΙ ΔΕΝ ΝΑ ΑΝΑΖΗΤΑ
• Γενικές ειδήσεις χωρίς συναισθηματικό φορτίο
• Διαφημίσεις, spam, bots
• Αθλητικά αποτελέσματα χωρίς πολιτική σύνδεση
• Περιεχόμενο εκτός χρονικού παραθύρου (default: 7 ημέρες)

═══════════════════════════════════════════════════════
2. ΠΩΣ ΝΑ ΑΝΑΖΗΤΑ
═══════════════════════════════════════════════════════

2.1 QUERY SET
| Τύπος | Παράδειγμα query | Σκοπός |
|---|---|---|
| Entity-only | "Κυριάκος Μητσοτάκης" OR "Μητσοτάκης" | Γενική ανίχνευση |
| Entity + θέμα | Μητσοτάκης AND (φόρος OR ακρίβεια) | Θεματικό sentiment |
| Sentiment trigger | "αδιέξοδο" OR "αίσχος" AND πολιτικ* | Φορτισμένο κείμενο |
| Δορυφορικές | [entity] AND (αντιδήμαρχος) AND (επεισόδιο) | Political spill detection |

2.2 ΦΙΛΤΡΑΡΙΣΜΑ
• Ελάχιστο μήκος: >= 20 λέξεις
• Γλωσσικό φίλτρο: el / en
• Keyword blacklist για διαφημίσεις
• Deduplication: hash-based
• Date filter: default 7 ημέρες

2.3 ΜΟΝΤΕΛΑ ΑΝΑΛΥΣΗΣ
Α — Lexicon-based: Greek Sentiment Lexicon | AFINN adapted for Greek | LIWC
Β — ML/Deep Learning: GreekBERT (Antypas et al., 2022) — κύριο μοντέλο | XLM-RoBERTa | LLM fallback: Claude

═══════════════════════════════════════════════════════
3. ΠΡΩΤΟΚΟΛΛΟ ΑΝΑΛΥΣΗΣ [v7.8]
═══════════════════════════════════════════════════════

ΚΑΝΟΝΑΣ: Μόλις ο χρήστης αναφέρει πολιτικό, ο agent ξεκινά ΑΜΕΣΑ αναζήτηση με web_search — χωρίς καμία ερώτηση.
- Χρονικό παράθυρο: Προεπιλογή = τελευταίος μήνας. Αν ο χρήστης ζητήσει διαφορετικό, το αλλάζεις.
- Θεματικές: Προεπιλογή = ΟΛΕΣ. Δεν ρωτάς ποτέ για θεματικές.
- Format παραδοτέου: ΠΑΝΤΑ απάντηση στο chat. Δεν ρωτάς ποτέ για format.
ΠΟΤΕ μην ζητάς διευκρινίσεις πριν ψάξεις. Ψάχνεις πρώτα, αναλύεις μετά.

3β. PUBLIC VOICE SIGNAL — Δημόσιες Φωνές [v7.7]
ΚΑΝΟΝΑΣ: Σε κάθε ανάλυση ο agent αναζητά αν κάποιος με δημόσιο βήμα — ή απλός πολίτης που έγινε viral — έχει μιλήσει για θέμα που ΣΥΣΧΕΤΙΖΕΤΑΙ με τον πολιτικό.

Κατηγορίες Δημόσιας Φωνής:
| Κατηγορία | Παραδείγματα | Γιατί έχει σημασία |
|---|---|---|
| Influencers/Content creators | Lifestyle, food, gaming creators | Μιλούν σε κοινό που δεν παρακολουθεί πολιτικά ΜΜΕ. |
| Δημοσιογράφοι (προσωπικά) | Μόνο εκτός επαγγελματικού ρόλου | Έχουν ακροατήριο και αξιοπιστία. |
| Αναλυτές/Ειδικοί | Οικονομολόγοι, νομικοί, γιατροί | Δίνουν κύρος ή εκθέτουν αδυναμίες. |
| Celebrities/Αθλητές | Ηθοποιοί, τραγουδιστές, αθλητές | Ο κόσμος ταυτίζεται μαζί τους. |
| Απλός πολίτης — viral post | Βίντεο/φωτογραφία που ανέλαβαν εθνικά ΜΜΕ | Αυθόρμητο και επαληθευμένο — ισχυρό signal. |

Output format Public Voice Signal:
- voice_entity: Όνομα / «Ανώνυμος πολίτης (viral)»
- voice_category: Influencer / Δημοσιογράφος / Αναλυτής / Celebrity / Viral πολίτης
- voice_statement: Τι ακριβώς είπε — 1-2 προτάσεις
- voice_sentiment: Θετικό / Αρνητικό / Ουδέτερο ως προς τον πολιτικό
- voice_reach: Followers ή «viral — κάλυψη από [ΜΜΕ]»
- contact_flag: ΑΞΙΖΕΙ ΕΠΑΦΗ: ΝΑΙ / ΟΧΙ

3γ. CIVIL SOCIETY SIGNAL — ΜΚΟ, Σύλλογοι & Φορείς [v7.7]
ΚΑΝΟΝΑΣ: Σε κάθε ανάλυση ο agent αναζητά αν ΜΚΟ, επαγγελματικοί σύλλογοι ή κοινωνικοί φορείς έχουν τοποθετηθεί δημόσια σε θέμα που ΣΥΣΧΕΤΙΖΕΤΑΙ με τον πολιτικό.

ΚΡΙΤΙΚΟΣ ΚΑΝΟΝΑΣ ΦΙΛΤΡΑΡΙΣΜΑΤΟΣ:
| Περίπτωση | Αποτέλεσμα |
|---|---|
| Φορέας τοποθετείται σε θέμα αρμοδιότητας πολιτικού | ΜΠΑΙΝΕΙ |
| Φορέας εκδίδει θέση για νόμο που κατέθεσε ο πολιτικός | ΜΠΑΙΝΕΙ |
| Φορέας βρίσκεται στις ειδήσεις για άσχετο θέμα | ΔΕΝ ΜΠΑΙΝΕΙ |
| Τοπικός φορέας σε θέμα δήμου/περιφέρειας | ΜΠΑΙΝΕΙ |

Κατηγορίες Φορέων:
| Κατηγορία | Παραδείγματα |
|---|---|
| Επαγγελματικοί Σύλλογοι | Ιατρικός Σύλλογος, Δικηγορικός Σύλλογος, ΤΕΕ, ΓΣΕΕ |
| ΜΚΟ — Κοινωνική Πολιτική | Médecins du Monde, Praksis, ARSIS, HumanRights360 |
| Περιβαλλοντικές Οργανώσεις | WWF Ελλάς, Greenpeace |
| Καταναλωτικές & Εργατικές | ΙΝΚΑ, ΕΚΠΟΙΖΩ, ΑΔΕΔΥ |
| Ακαδημαϊκά Σώματα | Σύνδεσμος Ελλήνων Ακαδημαϊκών |
| Τοπικοί Φορείς | Σύλλογοι Γονέων, Κατοίκων, Εμπορικοί Σύλλογοι |
| Διεθνείς Οργανισμοί | Amnesty International, HRW, UNHCR |

Output format Civil Society Signal:
- org_entity: Όνομα φορέα
- org_category: Τύπος φορέα
- org_statement: Τι ανακοίνωσε — 1-2 προτάσεις
- org_sentiment: Θετικό / Αρνητικό / Ουδέτερο
- org_reach: Εκτίμηση εμβέλειας
- org_channel: Δελτίο Τύπου / Ανακοίνωση / Social Media / Επιστολή
- org_bias_flag: ΑΜΕΡΟΛΗΠΤΟΣ ΦΟΡΕΑΣ / ΟΡΓΑΝΩΜΕΝΟ ΚΛΑΔΙΚΟ ΣΥΜΦΕΡΟΝ
- relevance_score: ΥΨΗΛΗ / ΜΕΣΑΙΑ / ΧΑΜΗΛΗ

═══════════════════════════════════════════════════════
4. OUTPUT — ΠΩΣ ΝΑ ΒΓΑΖΕΙ ΤΟ SENTIMENT
═══════════════════════════════════════════════════════

4.1 ΚΛΙΜΑΚΕΣ SENTIMENT
| Label | Ορισμός | Παράδειγμα |
|---|---|---|
| ΘΕΤΙΚΟ (+) | Υποστήριξη, έπαινος, αισιοδοξία. Score > +0.2 | Τα ΜΜΕ επαινούν. |
| ΑΡΝΗΤΙΚΟ (−) | Κριτική, οργή, σαρκασμός. Score < -0.2 | Τα ΜΜΕ κριτικάρουν. |
| ΟΥΔΕΤΕΡΟ (0) | Αντικειμενική αναφορά. -0.2 έως +0.2 | Απλή ανακοίνωση. |

4.1β POLITICAL_SPILL_RISK [ΝΕΟ v7]
Εκχύλιση αρνητικού sentiment από δορυφορική οντότητα προς τον κεντρικό πολιτικό. ΠΑΝΤΑ ξεχωριστά. Δεν συγχωνεύεται στο net score.

Output format:
- spill_entity: Δορυφορική οντότητα που πυροδότησε το spill
- spill_incident: Σύντομη τεχνική περιγραφή
- spill_count: Αριθμός δημόσιων αναφορών
- spill_risk_level: LOW / MEDIUM / HIGH
- net_score_impact: Διαφορά score πριν/μετά

═══════════════════════════════════════════════════════
5. WORKFLOW — ΒΗΜΑ-ΒΗΜΑ
═══════════════════════════════════════════════════════

1. Fetch νέων άρθρων/σχολίων (RSS + scraping) — κάθε 2h
2. Dedup + spam filter + lang detection
3. Preprocessing (normalize, tokenize, lemma)
4. Lexicon pass
5. ML model pass (GreekBERT / XLM-R)
6. ABSA — per-aspect scores
6b. Political Spill scan — δορυφορικές οντότητες
6γ. Civil Society Signal scan — ΜΚΟ, σύλλογοι [v7.7]
7. Irony detection + negation check
8. Aggregation ανά entity / θεματική
9. Alert check: spike detection
10. Weekly report generation — Κυριακή 08:00

═══════════════════════════════════════════════════════
6. ΠΡΩΤΟΚΟΛΛΟ ΛΙΓΩΝ ΔΕΔΟΜΕΝΩΝ [v7.3]
═══════════════════════════════════════════════════════

| Συνθήκη | Ορισμός | Ενέργεια |
|---|---|---|
| Πολύ χαμηλό volume | Κάτω από 3 δημοσιεύματα | Ρωτά αν θέλει εβδομαδιαία ανάλυση |
| Μεροληπτικές πηγές | 100% από μία ιδεολογική κατεύθυνση | Σημειώνει τον περιορισμό |
| Ελλιπής θεματική | Χωρίς επαρκή coverage | Ρωτά αν συμπεριλάβει προηγούμενη εβδομάδα |

═══════════════════════════════════════════════════════
7. BENCHMARKS [v7.3] — Ζώνες Score Εθνικών Πολιτικών
═══════════════════════════════════════════════════════

| Score | Ζώνη | Expert | Απλός Αναγνώστης |
|---|---|---|---|
| +0.6 έως +1.0 | Ισχυρή θετική | Momentum φάση. Σπάνιο για κυβερνητικό πολιτικό. | Εξαιρετική εβδομάδα. |
| +0.2 έως +0.59 | Μετρίως θετική | Φυσιολογική ζώνη. | Κυρίως καλά νέα. |
| -0.19 έως +0.19 | Ουδέτερη | Πολωμένη κάλυψη ή ήρεμη εβδομάδα. | Τα νέα είναι μοιρασμένα. |
| -0.2 έως -0.49 | Μετρίως αρνητική | Πίεση από ΜΜΕ και αντιπολίτευση. | Κυρίως αρνητικά νέα. |
| -0.5 έως -1.0 | Κρίση | Σοβαρό πρόβλημα δημόσιας εικόνας. | Πολύ άσχημη εβδομάδα. |

═══════════════════════════════════════════════════════
8. ΗΘΙΚΗ ΧΡΗΣΗ & ΠΕΡΙΟΡΙΣΜΟΙ
═══════════════════════════════════════════════════════

• Δεν αποθηκεύονται προσωπικά δεδομένα — μόνο aggregated scores
• Δεν χρησιμοποιείται για targeting ή manipulation πολιτών
• Τα δεδομένα διατηρούνται max 90 ημέρες και διαγράφονται
• Δεν αναλύονται accounts ανηλίκων
• Συμμόρφωση με GDPR και EU AI Act

═══════════════════════════════════════════════════════
ΟΔΗΓΙΕΣ ΛΕΙΤΟΥΡΓΙΑΣ
═══════════════════════════════════════════════════════

Απάντα ΠΑΝΤΑ στα Ελληνικά.
Πριν από κάθε ανάλυση χρησιμοποίησε ΥΠΟΧΡΕΩΤΙΚΑ το εργαλείο web_search για να αναζητήσεις πραγματικά, πρόσφατα δεδομένα από ελληνικά ΜΜΕ. Κάνε 2-3 αναζητήσεις με διαφορετικές λέξεις-κλειδιά (π.χ. όνομα πολιτικού, δηλώσεις, κόμμα). ΠΟΤΕ μην κάνεις ανάλυση χωρίς να έχεις πρώτα ψάξει.
Ακολούθα πάντα τη ΔΟΜΗ ΑΠΑΝΤΗΣΗΣ (6 βήματα) για κάθε ανάλυση.
Γράφε ΠΑΝΤΑ σε δύο επίπεδα: EXPERT + ΑΠΛΟΣ ΑΝΑΓΝΩΣΤΗΣ.
ΠΟΤΕ μην ζητάς από τον χρήστη να επιλέξει μεταξύ θεωρητικής ανάλυσης ή αναζήτησης — πάντα κάνε αναζήτηση αμέσως.
"""

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.6rem; }
    .main-header p { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.9rem; }
    .version-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-top: 0.5rem;
    }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📊 Political Sentiment Analysis Agent</h1>
    <p>EU Expert System για ανάλυση πολιτικού sentiment στα ελληνικά ΜΜΕ & social media</p>
    <span class="version-badge">Knowledge Base v7.7</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Ρυθμίσεις")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Εισάγετε το Anthropic API key σας. Δεν αποθηκεύεται."
    )

    st.markdown("---")
    st.markdown("### 📋 Τι κάνει ο Agent")
    st.markdown("""
    - 🔍 Αναλύει sentiment πολιτικών στα ΜΜΕ
    - 📊 Net Score -1.0 έως +1.0
    - ⚠️ Political Spill Risk detection
    - 🏛️ Civil Society Signal tracking
    - 📢 Public Voice Signal (influencers, viral)
    - 🎯 Δύο επίπεδα: Expert + Απλός Αναγνώστης
    """)

    st.markdown("---")
    st.markdown("### 💡 Παραδείγματα ερωτημάτων")
    example_queries = [
        "Ανάλυσε το sentiment για τον Κυριάκο Μητσοτάκη",
        "Τι λένε τα ΜΜΕ για τον Νίκο Ανδρουλάκη;",
        "Ανάλυση sentiment για τον Δήμαρχο Αθηναίων",
        "Εξήγησέ μου τι είναι το Political Spill Risk",
    ]
    for q in example_queries:
        if st.button(q, use_container_width=True):
            st.session_state.pending_input = q

    if st.button("🗑️ Καθαρισμός συνομιλίας", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

# ─────────────────────────────────────────────
# API KEY RESOLUTION
# ─────────────────────────────────────────────
def get_api_key():
    # Priority: sidebar input > streamlit secrets
    if api_key:
        return api_key
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

# ─────────────────────────────────────────────
# WELCOME MESSAGE
# ─────────────────────────────────────────────
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "👋 **Καλωσήρθατε στο Political Sentiment Analysis Agent (v7.7)**\n\n"
            "Είμαι ένας εξειδικευμένος AI agent για την ανάλυση πολιτικού sentiment στα ελληνικά ΜΜΕ και social media.\n\n"
            "**Τι μπορώ να κάνω:**\n"
            "- 📊 Ανάλυση sentiment πολιτικών (Net Score -1.0 έως +1.0)\n"
            "- ⚠️ Εντοπισμός Political Spill Risk από δορυφορικές οντότητες\n"
            "- 🏛️ Civil Society Signal — ΜΚΟ, σύλλογοι, φορείς\n"
            "- 📢 Public Voice Signal — influencers, ειδικοί, viral πολίτες\n"
            "- 🎯 Δύο επίπεδα ανάλυσης: Expert + Απλός Αναγνώστης\n\n"
            "**Πείτε μου ποιον πολιτικό θέλετε να αναλύσω** και αναζητώ αμέσως στα ελληνικά ΜΜΕ."
        )
    })

# ─────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ─────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─────────────────────────────────────────────
# HANDLE INPUT
# ─────────────────────────────────────────────
user_input = st.chat_input("Γράψτε το ερώτημά σας εδώ...")

# Handle sidebar example button clicks
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None

if user_input:
    resolved_key = get_api_key()
    if not resolved_key:
        st.error("⚠️ Δεν βρέθηκε Anthropic API key. Εισάγετε το στο sidebar.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ── Web search tool definition ──────────────────────────────────────────
    WEB_SEARCH_TOOL = {
        "name": "web_search",
        "description": (
            "Ψάχνει σε ελληνικά ΜΜΕ και news sites για πληροφορίες σχετικές με "
            "ελληνικούς πολιτικούς, κόμματα, δηλώσεις και πολιτικά γεγονότα. "
            "Χρησιμοποίησε αυτό το tool για να βρεις πραγματικές, πρόσφατες ειδήσεις "
            "πριν κάνεις την ανάλυση sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Η λέξη-κλειδί ή φράση αναζήτησης (ελληνικά ή αγγλικά)"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Αριθμός αποτελεσμάτων (1-10, default 8)",
                    "default": 8
                }
            },
            "required": ["query"]
        }
    }

    def do_web_search(query: str, num_results: int = 8) -> str:
        import requests
        import xml.etree.ElementTree as ET
        import html
        import re

        # Greek news RSS feeds
        RSS_FEEDS = [
            ("Καθημερινή",    "https://www.kathimerini.gr/rss"),
            ("Proto Thema",   "https://www.protothema.gr/rss/"),
            ("iefimerida",    "https://www.iefimerida.gr/feed"),
            ("in.gr",         "https://www.in.gr/feed/"),
            ("news247",       "https://news247.gr/feed/"),
            ("Documento",     "https://www.documentonews.gr/feed/"),
            ("Liberal",       "https://www.liberal.gr/feed/"),
            ("Τα Νέα",        "https://www.tanea.gr/feed/"),
            ("Εφημερίδα Συντακτών", "https://www.efsyn.gr/rss.xml"),
            ("Avgi",          "https://www.avgi.gr/rss.xml"),
            ("reporter.gr",   "https://www.reporter.gr/rss/"),
            ("CNN Greece",    "https://www.cnn.gr/rss"),
        ]

        # Normalize query words for matching
        def normalize(text):
            return re.sub(r'[^\w\s]', '', text.lower())

        q_words = [w for w in normalize(query).split() if len(w) > 2]

        def strip_html(text):
            text = re.sub(r'<[^>]+>', ' ', text or '')
            return html.unescape(text).strip()

        all_hits = []

        for source, url in RSS_FEEDS:
            try:
                r = requests.get(url, timeout=6,
                                 headers={"User-Agent": "Mozilla/5.0 (compatible; NewsBot/1.0)"})
                if r.status_code != 200:
                    continue
                root = ET.fromstring(r.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}
                items = root.findall('.//item') + root.findall('.//atom:entry', ns)
                for item in items:
                    title_el = item.find('title') or item.find('atom:title', ns)
                    desc_el  = item.find('description') or item.find('atom:summary', ns) or item.find('atom:content', ns)
                    link_el  = item.find('link') or item.find('atom:link', ns)

                    title = strip_html(title_el.text if title_el is not None else '')
                    desc  = strip_html(desc_el.text  if desc_el  is not None else '')
                    link  = (link_el.get('href') if link_el is not None and link_el.get('href')
                             else (link_el.text if link_el is not None else ''))

                    combined = normalize(title + ' ' + desc)
                    score = sum(1 for w in q_words if w in combined)
                    if score > 0:
                        all_hits.append((score, source, title, link, desc[:400]))
            except Exception:
                continue

        # Sort by relevance score
        all_hits.sort(key=lambda x: -x[0])

        if all_hits:
            lines = []
            for i, (score, source, title, link, body) in enumerate(all_hits[:num_results], 1):
                lines.append(f"[{i}] [{source}] {title}\n{link}\n{body}")
            return "\n\n".join(lines)

        # Fallback: try DuckDuckGo
        try:
            import time
            time.sleep(1)
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results))
            if results:
                lines = []
                for i, r in enumerate(results, 1):
                    lines.append(f"[{i}] {r.get('title','')}\n{r.get('href','')}\n{r.get('body','')}")
                return "\n\n".join(lines)
        except Exception:
            pass

        return f"Δεν βρέθηκαν αποτελέσματα για: «{query}». Δοκιμάστε διαφορετικές λέξεις-κλειδιά."

    # Call Claude API with agentic loop
    with st.chat_message("assistant"):
        try:
            client = anthropic.Anthropic(api_key=resolved_key)

            api_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ]

            response = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=[WEB_SEARCH_TOOL],
                messages=api_messages,
            )

            # Agentic loop — handles ALL tool_use blocks in each response
            while response.stop_reason == "tool_use":
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                tool_results = []

                for tool_use_block in tool_use_blocks:
                    query = tool_use_block.input.get("query", "")
                    num_res = tool_use_block.input.get("num_results", 8)

                    with st.spinner(f"🔍 Αναζητώ: «{query}»..."):
                        search_result = do_web_search(query, num_res)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": search_result
                    })

                # Append assistant turn + ALL tool results in one user turn
                api_messages.append({"role": "assistant", "content": response.content})
                api_messages.append({
                    "role": "user",
                    "content": tool_results
                })

                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=[WEB_SEARCH_TOOL],
                    messages=api_messages,
                )

            assistant_reply = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            st.markdown(assistant_reply)
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

        except anthropic.AuthenticationError:
            st.error("❌ Λάθος API key. Ελέγξτε και δοκιμάστε ξανά.")
        except anthropic.RateLimitError:
            st.error("⏳ Rate limit. Δοκιμάστε ξανά σε λίγα δευτερόλεπτα.")
        except Exception as e:
            st.error(f"❌ Σφάλμα: {str(e)}")
