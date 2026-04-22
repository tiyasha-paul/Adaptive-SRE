# PROGRESS.md — AdaptiveSRE Build Status

Last updated: 2026-04-22
Current phase: 2

## Completed phases
- [x] Phase 0 — Init
- [x] Phase 1 — Mock services
- [x] Phase 2 — Models + service graph
- [ ] Phase 3 — Lead engineer + fault injector + docker executor
- [ ] Phase 4 — Grader
- [ ] Phase 5 — Environment core
- [ ] Phase 6 — FastAPI server + Gradio UI
- [ ] Phase 7 — inference.py
- [ ] Phase 8 — openenv.yaml + Dockerfile
- [ ] Phase 9 — Training pipeline
- [ ] Phase 10 — Full validation

## Files created (fill as built)
- AGENT.md
- MASTER_BUILD_GUIDE.md
- requirements.txt
- mock_services/db/main.py, Dockerfile
- mock_services/auth/main.py, Dockerfile
- mock_services/payment/main.py, Dockerfile
- mock_services/cache/main.py, Dockerfile
- mock_services/notification/main.py, Dockerfile
- mock_services/docker-compose.yml
- server/__init__.py
- server/models.py
- server/service_graph.py

## Decisions that deviate from AGENT.md
- DB port changed from 5432 to 15432 (local PostgreSQL uses 5432)

## Measured results (fill from actual runs)
Gen 0 mean reward (easy): TBD
Gen 0 mean reward (medium): TBD
Gen 0 mean reward (hard): TBD
Gen 1 mean reward (easy): TBD

## Next step
Phase 3 — Lead engineer + fault injector + docker executor
