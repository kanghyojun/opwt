# AGENTS.md

이 문서는 이 저장소에서 작업하는 코딩 에이전트용 작업 가이드입니다.

## Project Snapshot

- 앱: `opwt.py` 단일 파일 기반 curses TUI
- 목적: Git worktree 상태와 OpenCode 세션 상태를 한 화면에서 확인
- 주요 상태: `idle`, `running`, `permission-ask`

## Common Commands

- 실행: `python3 opwt.py [path]`
- 빌드: `./build.sh`
- 최소 문법 검증: `python3 -m py_compile opwt.py`

## Working Rules

- 기본적으로 기존 아키텍처(단일 파일, 표준 라이브러리 중심)를 유지합니다.
- 사용자 요청이 없는 한 새 의존성 추가, 대규모 리팩터링, 키바인딩 변경을 하지 않습니다.
- Git/프로세스 감지 로직 수정 시 타임아웃, 예외 메시지, fallback 동작을 유지합니다.
- `/proc` 기반 세션 추론은 휴리스틱이므로 보수적으로 변경합니다.

## Validation Checklist

- 앱이 정상 실행되는지 확인: `python3 opwt.py`
- Python 문법 확인: `python3 -m py_compile opwt.py`
- 빌드 변경 시 결과물 확인: `./build.sh` 후 `dist/opwt`

## Scope Notes

- Alias 저장 경로: `~/.config/opwt/aliases.json`
- OpenCode DB 경로: `~/.local/share/opencode/opencode.db` (환경 변수에 따라 달라질 수 있음)
