> Handoff temporaneo per la migrazione preparata il 9 novembre 2026, pubblicato il 13 novembre 2026. Questa cartella contiene solo istruzioni e checksum: l'archivio con dati e risultati non e' in GitHub.
>
> Sulla macchina originale l'archivio e il sidecar sono in `/home/marco/projects/graph_recon/migration/`, accanto alla cartella `W-DGM`. Copiare entrambi sulla destinazione. Il checksum e' incluso anche qui.
>
> Dopo il trasferimento e i controlli sulla nuova macchina, questa intera cartella puo' essere eliminata insieme al relativo richiamo nel README. Mantenere la guida generale `docs/MIGRATION.md` e gli strumenti di migrazione. Conservare l'archivio originale fino alla verifica del backup.
>
> Il commit indicato sotto identifica lo snapshot nell'archivio, non l'ultimo commit della repo. Il checkout di quel commit e' intenzionale; questa guida, aggiunta successivamente, rimane consultabile su GitHub. I comandi si eseguono dalla radice del progetto, non da questa cartella. `RESTORE.md` si trova in `migration/RESTORE.md` dopo l'estrazione.

# Trasferimento del workspace del paper

Repository: https://github.com/MarcoCasalbore/Graph-water-recon
Commit esatto da ripristinare: 44c4fa88d09f3f6c0291fb74028ab1cb3b86e8f2
Archivio: graph-water-recon-paper-2026-11-09.tar.gz
Origine: /home/marco/projects/graph_recon/W-DGM

Il pacchetto contiene codice/config/test/notebook, dataset e fonti selezionate, risultati finali Net3/Iowa e baseline, sintetico e ablation, otto reti supplementari, figure/manoscritto/PPTX e bibliografia. Non contiene WDGM, credenziali, .git o i dati originali Rieti/CSA di other_research.

1. Copiare archivio e sidecar .sha256 sulla nuova macchina. Verificare SHA-256 prima dell'estrazione.
2. Clonare Graph-water-recon in una cartella nuova e fare git checkout 44c4fa88d09f3f6c0291fb74028ab1cb3b86e8f2. In alternativa, l'archivio contiene anche uno snapshot dei sorgenti e si può usare senza Git.
3. Dalla cartella destinazione: tar -xzf PERCORSO_ARCHIVIO --strip-components=1 -C .
4. Con Python disponibile: python scripts/maintenance/migrate_workspace.py verify
5. Anteprima: python scripts/maintenance/migrate_workspace.py rebase
6. Applicare: python scripts/maintenance/migrate_workspace.py rebase --apply
7. Verificare: python scripts/maintenance/migrate_workspace.py verify --manifest migration/manifest-relocated.json
8. Ricreare WDGM con Python 3.11, installare il PyTorch adeguato alla macchina e poi requirements.txt. I comandi completi sono in ../docs/MIGRATION.md. Il vecchio WDGM resta funzionante sulla macchina originale.
9. Sulla nuova macchina eseguire pip check, test/smoke CPU e poi GPU esplicita se disponibile; importare i simulatori prima delle campagne. Non avviare automaticamente tutte le ablation.

Configurazione più vicina alla sorgente: Linux/WSL2 + NVIDIA + Torch 2.8.0/cu129. Windows CPU è stato provato dalla CI per il percorso principale; Windows GPU e AMD/ROCm richiedono verifica sulla destinazione. Per nomi storici lunghi usare Linux/WSL o abilitare i percorsi lunghi Windows.

Il freeze dell'ambiente è documentazione della sorgente, non un installer universale. Le versioni backend/hardware possono cambiare i risultati numerici. Le vecchie campagne senza run_identity restano risultati storici: per nuovi training scegliere nuove cartelle outputs/ e non aggirare i controlli di resume.

Le prove del modello non sono state rieseguite preparando questo pacchetto. Sono stati controllati il trasferimento/hash e lo strumento di adattamento dei percorsi con test dedicati. Per il significato di esperimenti e figure leggere ../docs/PAPER_REPRODUCTION.md e ../docs/STATUS.md. I notebook dei grafi includono snapshot di layout; per nuovi grafi usare gli esportatori del run.

SHA-256 archivio verificato: 95378de81308010fe3a9e8377dfb7f4769ee380265d0f613cbbe0a6e30fbf298

File nel pacchetto: 31383. Contenuto non compresso: 1.061 GiB.

Stato della consegna
-------------------
- Tutti i 31383 file dell'archivio sono stati letti e verificati contro il manifest, senza estrarre copie aggiuntive del dataset.
- Dimensione archivio: 790,975,207 byte (circa 791.0 MB); payload: 1.061 GiB.
- Il comando di ricollocazione è stato verificato con 5 test dedicati (integrità, percorsi non sicuri, rollback, conservazione dei numeri), non con un nuovo training.
- Su 2261 JSON reali è stata provata in memoria la sostituzione verso una destinazione fittizia, confermando che il testo esterno alle stringhe resta identico. I file originali non sono stati modificati.
- Percorso relativo più lungo nel pacchetto: 189 caratteri. Su Windows scegliere una cartella corta o abilitare i percorsi lunghi.
- La destinazione non è stata ancora configurata: compatibilità GPU, simulatori e nuovo ambiente vanno controllati lì prima delle campagne scientifiche.
- L'archivio è un trasferimento locale: non è caricato su GitHub. Nella repo sono pubblicati solo codice, configurazioni, notebook sorgente e istruzioni.

Prossimo passo operativo: trasferire .tar.gz, .sha256 e questo handoff, poi seguire RESTORE.md/../docs/MIGRATION.md. Nessuna campagna costosa va lanciata automaticamente.
