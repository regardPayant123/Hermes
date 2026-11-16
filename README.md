# Hermes
A real version of jarvis

JARVIS/
│
├── apps/
│   ├── desktop/
│   ├── mobile/
│   └── embedded/
│
├── backend/
│   ├── gateway/
│   ├── orchestrator/
│   ├── authentication/
│   └── event_bus/
│
├── engines/
│   ├── voice/
│   │   ├── wake_word/
│   │   ├── vad/
│   │   ├── stt/
│   │   └── tts/
│   │
│   ├── brain/
│   │   ├── llm/
│   │   ├── planner/
│   │   └── tool_router/
│   │
│   ├── memory/
│   │   ├── short_term/
│   │   ├── long_term/
│   │   └── semantic/
│   │
│   ├── vision/
│   └── actions/
│
├── cpp/
│   ├── audio/
│   ├── vision/
│   └── inference/
│
├── infrastructure/
│   ├── docker/
│   ├── postgres/
│   ├── redis/
│   └── monitoring/
│
├── training/
│   ├── datasets/
│   ├── models/
│   └── experiments/
│
├── tests/
├── configs/
└── main.py


