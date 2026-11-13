# Handoff temporaneo: workspace completo del paper

Archivio e checksum sono disponibili nella [Release di questa repo](https://github.com/MarcoCasalbore/Graph-water-recon/releases/tag/paper-workspace-2026-11-09). Non serve accedere al vecchio PC. Il clone fornisce il codice; gli allegati della Release forniscono il workspace conservato del paper. GitHub limita i file Git ordinari a 100 MiB, quindi l'archivio da 791 MB e' distribuito come Release asset.

Download Linux/WSL (da una cartella destinata ai download):

```bash
curl -fL --retry 3 -O https://github.com/MarcoCasalbore/Graph-water-recon/releases/download/paper-workspace-2026-11-09/graph-water-recon-paper-2026-11-09.tar.gz
curl -fL --retry 3 -O https://github.com/MarcoCasalbore/Graph-water-recon/releases/download/paper-workspace-2026-11-09/graph-water-recon-paper-2026-11-09.tar.gz.sha256
sha256sum -c graph-water-recon-paper-2026-11-09.tar.gz.sha256
```

In Windows PowerShell usare `curl.exe` al posto di `curl` nei due comandi di download, poi verificare:

```powershell
(Get-FileHash .\graph-water-recon-paper-2026-11-09.tar.gz -Algorithm SHA256).Hash -eq '95378de81308010fe3a9e8377dfb7f4769ee380265d0f613cbbe0a6e30fbf298'
```

Procedere solo se il checksum corrisponde (`OK` o `True`). Seguire poi i passi di ripristino sotto e la [guida generale](../docs/MIGRATION.md), inclusi ambiente, test CPU/GPU e simulatori. Python 3.11, Git, connessione Internet e spazio per archivio, estrazione e ambiente sono prerequisiti; l'ambiente binario della vecchia macchina non e' portabile.

Il commit sotto identifica lo snapshot **dentro l'archivio**, non l'ultimo commit della repo. Il checkout di quel commit e' intenzionale per verificare tutti gli hash dello snapshot. Questa guida, aggiunta successivamente, rimane consultabile su GitHub. Eseguire i comandi dalla radice del progetto; `RESTORE.md` e' in `migration/RESTORE.md` dopo l'estrazione.

Questa cartella e il suo richiamo nel README possono essere eliminati dopo trasferimento, controlli e backup verificato. Conservare la Release (o un suo backup verificato), la guida generale e gli strumenti di migrazione: cancellare l'unica copia dell'archivio eliminerebbe l'accesso ai risultati conservati.

---

# Trasferimento del workspace del paper

Repository: https://github.com/MarcoCasalbore/Graph-water-recon
Commit esatto da ripristinare: 44c4fa88d09f3f6c0291fb74028ab1cb3b86e8f2
Archivio: graph-water-recon-paper-2026-11-09.tar.gz
Origine: /home/marco/projects/graph_recon/W-DGM

Il pacchetto contiene codice/config/test/notebook, dataset e fonti selezionate, risultati finali Net3/Iowa e baseline, sintetico e ablation, otto reti supplementari, figure/manoscritto/PPTX e bibliografia. Non contiene WDGM, credenziali, .git o i dati originali Rieti/CSA di other_research.

1. Scaricare archivio e sidecar .sha256 dalla Release indicata qui sopra. Verificare SHA-256 prima dell'estrazione.
2. Clonare Graph-water-recon in una cartella nuova e fare git checkout 44c4fa88d09f3f6c0291fb74028ab1cb3b86e8f2. In alternativa, l'archivio contiene anche uno snapshot dei sorgenti e si può usare senza Git.
3. Dalla cartella destinazione: tar -xzf PERCORSO_ARCHIVIO --strip-components=1 -C .
4. Con Python disponibile: python scripts/maintenance/migrate_workspace.py verify
5. Anteprima: python scripts/maintenance/migrate_workspace.py rebase
6. Applicare: python scripts/maintenance/migrate_workspace.py rebase --apply
7. Verificare: python scripts/maintenance/migrate_workspace.py verify --manifest migration/manifest-relocated.json
8. Dopo entrambe le verifiche, eseguire `git switch main` nel clone per tornare al codice mantenuto (i dati/risultati ignorati restano al loro posto). I manifest descrivono lo snapshot storico e vanno verificati prima di questo cambio. Ricreare WDGM con Python 3.11, installare il PyTorch adeguato alla macchina, eseguire `python -m pip install six==1.17.0` e poi installare requirements.txt. Il passo `six` corregge una dipendenza non dichiarata di WNTR nello snapshot congelato; i requisiti di main includono gia la correzione. I comandi completi sono in ../docs/MIGRATION.md. Il vecchio WDGM resta funzionante sulla macchina originale.
9. Sulla nuova macchina eseguire pip check, test/smoke CPU e poi GPU esplicita se disponibile; importare i simulatori prima delle campagne. Non avviare automaticamente tutte le ablation.

Configurazione più vicina alla sorgente: Linux/WSL2 + NVIDIA + Torch 2.8.0/cu129. Windows CPU è stato provato dalla CI per il percorso principale; Windows GPU e AMD/ROCm richiedono verifica sulla destinazione. Per nomi storici lunghi usare Linux/WSL o abilitare i percorsi lunghi Windows.

Il freeze dell'ambiente è documentazione della sorgente, non un installer universale. Le versioni backend/hardware possono cambiare i risultati numerici. Le vecchie campagne senza run_identity restano risultati storici: per nuovi training scegliere nuove cartelle outputs/ e non aggirare i controlli di resume.

Le campagne scientifiche complete non sono state rieseguite preparando questo pacchetto; i controlli software del ripristino sono riportati in fondo. Sono stati controllati il trasferimento/hash e lo strumento di adattamento dei percorsi con test dedicati. Per il significato di esperimenti e figure leggere ../docs/PAPER_REPRODUCTION.md e ../docs/STATUS.md. I notebook dei grafi includono snapshot di layout; per nuovi grafi usare gli esportatori del run.

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
- Archivio e checksum sono pubblicati come allegati della Release di questa repo. Non sono inclusi nel clone Git o nello ZIP automatico dei sorgenti: scaricarli con i comandi qui sopra.

Prossimo passo operativo: trasferire .tar.gz, .sha256 e questo handoff, poi seguire RESTORE.md/../docs/MIGRATION.md. Nessuna campagna costosa va lanciata automaticamente.

## Controlli di ripristino del 13 novembre 2026

- Estrazione in una cartella nuova; verifica di tutti i 31.383 file prima della ricollocazione e del manifest aggiornato dopo la modifica di 2.272 JSON/CSV.
- Ambiente Python 3.11 CPU creato da zero, installazione dei requisiti completi con `six==1.17.0`, `pip check` superato.
- Tutti i 15 test software superati; import di WNTR, OpenDSSDirect, pandapower, SimBench e PyTorch Geometric riusciti.
- Download completo dell'archivio pubblicato su GitHub: SHA-256 identico all'originale locale.
- Smoke end to end sia water sia electric: generazione fixture, training di un'epoca, selezione, consenso, decodifica e valutazione riusciti.
- Nessuna dipendenza dal WDGM originale per questi controlli dell'ambiente nuovo. Le fixture sono controlli software, non nuovi risultati scientifici del paper.

Le campagne scientifiche complete, tutte le figure/tabelle e le GPU della destinazione non sono state rieseguite. La riproducibilita' numerica esatta tra hardware/backend diversi non e' garantita. Seguire la mappa `docs/PAPER_REPRODUCTION.md` per distinguere i protocolli sintetico e benchmark.
