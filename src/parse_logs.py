import re
import os

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURA QUI (Struttura a Tuple: (path_file, tag_modalita))
# ══════════════════════════════════════════════════════════════════════════════

LOG_FILES = [
    # ─── BABY (Multimodali Standard -> T+I) ───────────────────────────────────
    ('../log/baby/baby_BPR/BPR-baby-Mar-11-2026-21-31-01.log', "T+I"),
    ('../log/baby/baby_VBPR/VBPR-baby-Mar-11-2026-16-51-37.log', "T+I"),
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-Mar-18-2026-00-30-38.log', "T+I"),
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-Mar-19-2026-19-26-36.log', "T+I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-Mar-25-2026-18-07-08.log', "T+I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-Mar-28-2026-03-25-29.log', "T+I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-Mar-29-2026-14-09-26.log', "T+I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-Mar-29-2026-18-57-28.log', "T+I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-Mar-31-2026-04-17-19.log', "T+I"),
    
    # ─── SPORTS (Multimodali Standard -> T+I) ─────────────────────────────────
    ('../log/sports/sports_BPR/BPR-sports-Mar-13-2026-12-33-48.log', "T+I"),
    ('../log/sports/sports_VBPR/VBPR-sports-Mar-13-2026-18-21-05.log', "T+I"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-Mar-20-2026-01-44-45.log', "T+I"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-Mar-21-2026-03-10-35.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-01-2026-04-02-59.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-03-2026-07-48-36.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-04-2026-10-40-41.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-05-2026-12-28-36.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-07-2026-03-26-47.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-08-2026-14-31-24.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-09-2026-20-26-15.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-11-2026-00-22-51.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-11-2026-10-56-45.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-12-2026-02-55-26.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-13-2026-00-05-15.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-13-2026-15-10-03.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-14-2026-08-48-55.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-16-2026-04-38-33.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-17-2026-08-46-33.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-18-2026-09-30-10.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-18-2026-21-02-25.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-28-2026-11-29-02.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-28-2026-13-57-00.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-29-2026-03-37-34.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-29-2026-10-47-23.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Apr-29-2026-18-37-04.log', "T+I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-May-01-2026-14-46-53.log', "T+I"),
    
    # ─── MOVIELENS_1M (Multimodali Standard -> T+I) ───────────────────────────
    ('../log/movielens_1m/movielens_1m_BPR/BPR-movielens_1m-Mar-21-2026-17-32-01.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_VBPR/VBPR-movielens_1m-Mar-21-2026-13-04-35.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-Mar-21-2026-22-50-02.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-Mar-23-2026-02-33-11.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-02-2026-02-26-16.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-03-2026-03-35-30.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-03-2026-11-53-57.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-03-2026-20-30-57.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-04-2026-10-00-16.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-05-2026-20-51-24.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-06-2026-02-31-47.log', "T+I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-May-07-2026-07-38-19.log', "T+I"),
    
    # ─── SINGOLE MODALITÀ (Configurate ad hoc) ────────────────────────────────
    #baby
    ('../log/baby/baby_VBPR/VBPR-baby-May-08-2026-02-02-35.log', "I"),
    ('../log/baby/baby_VBPR/VBPR-baby-May-08-2026-23-14-50.log', "T"),
    
    # Modificare i tag sottostanti in base alla singola modalità reale dei log di FREEDOM baby (es. "T" o "I")
    # Per ora ho inserito dei valori d'esempio (T o I), configurali esattamente come preferisci:
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-May-14-2026-14-33-10.log', "T"),
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-May-15-2026-19-21-05.log', "T"),
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-May-16-2026-03-09-06.log', "I"),
    ('../log/baby/baby_FREEDOM/FREEDOM-baby-May-17-2026-02-13-47.log', "I"),
    
    # LATTICE - Baby (Aggiunti)
    ('../log/baby/baby_LATTICE/LATTICE-baby-May-26-2026-16-53-37.log', "I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-May-28-2026-02-10-38.log', "I"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-May-30-2026-00-07-23.log', "T"),
    ('../log/baby/baby_LATTICE/LATTICE-baby-May-31-2026-03-20-39.log', "T"),
    
    
    #sports
    ('../log/sports/sports_VBPR/VBPR-sports-May-11-2026-12-56-16.log', "I"),
    ('../log/sports/sports_VBPR/VBPR-sports-May-12-2026-05-52-07.log', "T"),
    
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-17-2026-10-32-37.log', "T"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-18-2026-14-10-41.log', "T"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-18-2026-19-31-58.log', "T"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-19-2026-01-38-01.log', "T"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-19-2026-13-26-20.log', "I"),
    ('../log/sports/sports_FREEDOM/FREEDOM-sports-May-20-2026-22-41-21.log', "I"),
    
    # LATTICE - Sports (Aggiunti)
    ('../log/sports/sports_LATTICE/LATTICE-sports-May-31-2026-21-29-39.log', "I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-02-2026-02-39-35.log', "I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-03-2026-02-40-11.log', "I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-04-2026-08-53-04.log', "I"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-06-2026-03-05-05.log', "T"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-07-2026-06-35-46.log', "T"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-08-2026-03-31-22.log', "T"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-09-2026-16-01-21.log', "T"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-11-2026-03-40-20.log', "T"),
    ('../log/sports/sports_LATTICE/LATTICE-sports-Jun-12-2026-07-43-33.log', "T"),
    
    
    
    #movielens_1m
    ('../log/movielens_1m/movielens_1m_VBPR/VBPR-movielens_1m-May-13-2026-01-37-21.log', "I"),
    ('../log/movielens_1m/movielens_1m_VBPR/VBPR-movielens_1m-May-13-2026-22-42-48.log', "T"),
    
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-May-22-2026-04-28-27.log', "T"),
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-May-23-2026-01-46-33.log', "T"),
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-May-23-2026-17-30-51.log', "I"),
    ('../log/movielens_1m/movielens_1m_FREEDOM/FREEDOM-movielens_1m-May-24-2026-10-42-15.log', "I"),
    
    # LATTICE - Movielens_1m (Aggiunti)
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-Jun-13-2026-10-17-39.log', "T"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-Jun-14-2026-12-06-14.log', "T"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-Jun-15-2026-18-03-46.log', "I"),
    ('../log/movielens_1m/movielens_1m_LATTICE/LATTICE-movielens_1m-Jun-17-2026-11-32-39.log', "I"),
]

OUTPUT_FILE = 'mmrec_summary.tsv'

# ══════════════════════════════════════════════════════════════════════════════


def extract_table(lines, start_idx):
    """Estrae i dati della tabella @k 5, 10, 20, 50"""
    results = {}
    re_row = re.compile(r"(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")
    for i in range(start_idx, start_idx + 10):
        if i >= len(lines): break
        m = re_row.search(lines[i])
        if m:
            k = int(m.group(1))
            results[k] = [m.group(2), m.group(3), m.group(4), m.group(5)]
    return results

def parse_log(file_path, modality_tag):
    all_runs = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    base_name = os.path.basename(file_path)
    parts = base_name.split('-')
    model_name = parts[0] if len(parts) > 0 else "Model"
    dataset_name = parts[1] if len(parts) > 1 else "Dataset"

    # Essendo che il tag deve esserci sempre, formattiamo con la virgola prima del tag passato
    mod_string = f", {modality_tag}"

    current_params = None

    for i, line in enumerate(lines):
        # 1. Cattura i parametri correnti
        p_match = re.search(r"Parameters:.*=\((.*)\)======", line)
        if p_match:
            current_params = p_match.group(1)
            continue

        # 2. Quando finisce il training, cattura tutto il blocco finale
        if "Finished training, best eval result in epoch" in line:
            # Associa parametri e tag all'interno delle parentesi tonde
            run_id = f"{model_name}-{dataset_name}({current_params}{mod_string})"
            
            data = {
                'id': run_id,
                'energies': {},
                'valid': {},
                'test': {}
            }
            
            # Cerca nelle 30 righe successive al trigger per estrarre energie e metriche
            for j in range(i, i + 30):
                if j >= len(lines): break
                l = lines[j]
                
                # Estrazione Energie
                if "TOTAL_TRAIN_ENERGY=" in l: data['energies']['train'] = l.split('=')[1].split()[0]
                if "TOTAL_VALID_ENERGY=" in l: data['energies']['valid'] = l.split('=')[1].split()[0]
                if "TOTAL_TEST_ENERGY=" in l:  data['energies']['test'] = l.split('=')[1].split()[0]
                if "TOTAL_ENERGY=" in l:       data['energies']['total'] = l.split('=')[1].split()[0]
                
                # Estrazione Tabelle (Best Valid e Test)
                if "best valid result:" in l:
                    data['valid'] = extract_table(lines, j + 1)
                if "test result:" in l:
                    data['test'] = extract_table(lines, j + 1)
            
            all_runs.append(data)
    
    return all_runs

def main():
    final_results = []
    for f, modality in LOG_FILES:
        if os.path.exists(f):
            final_results.extend(parse_log(f, modality))
        else:
            print(f"Attenzione: file non trovato -> {f}")

    if not final_results:
        print("Nessun blocco 'Finished training' trovato. Controlla i file di log.")
        return

    ks = [5, 10, 20, 50]
    metrics = ['map', 'ndcg', 'prec', 'recall']
    header = ['run_id', 'train_energy', 'valid_energy', 'test_energy', 'total_energy']
    for stage in ['v', 't']:
        for k in ks:
            for m in metrics:
                header.append(f'{stage}_{m}@{k}')

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        out.write('\t'.join(header) + '\n')
        for run in final_results:
            e = run['energies']
            row = [
                run['id'], 
                e.get('train', ''), e.get('valid', ''), 
                e.get('test', ''), e.get('total', '')
            ]
            for stage in ['valid', 'test']:
                for k in ks:
                    row.extend(run[stage].get(k, ['', '', '', '']))
            out.write('\t'.join(row) + '\n')
    
    print(f"Creato {OUTPUT_FILE} con {len(final_results)} run.")

if __name__ == "__main__":
    main()