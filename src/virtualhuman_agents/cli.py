from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import uvicorn

from .demo import run_demo
from .regulatory import RegulatoryService
from .store import SQLiteStore


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid command-line arguments\n")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="virtualhuman-agent",
        description="Auditable AI-agent platform for human disease-model R&D",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a two-site locked-validation demonstration")
    demo.add_argument("--db", default="virtualhuman-demo.db")
    demo.add_argument("--package-root", default="regulatory_packages")

    subparsers.add_parser(
        "humanfm-demo",
        help="run a synthetic forward pass through the cross-scale HumanFM reference model",
    )

    serve = subparsers.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--db", default="virtualhuman.db")
    serve.add_argument("--package-root", default="regulatory_packages")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    status = subparsers.add_parser("status", help="show project and regulatory readiness")
    status.add_argument("project_id")
    status.add_argument("--db", default="virtualhuman.db")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "demo":
        summary = run_demo(Path(args.db), Path(args.package_root))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.command == "humanfm-demo":
        from .foundation.demo import run_humanfm_demo

        print(json.dumps(run_humanfm_demo(), ensure_ascii=False, indent=2))
        return
    if args.command == "serve":
        if args.host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("legacy research API may bind only to a loopback host")
        from .api import create_app

        uvicorn.run(
            create_app(args.db, args.package_root),
            host=args.host,
            port=args.port,
            reload=False,
        )
        return
    if args.command == "status":
        store = SQLiteStore(args.db)
        project = store.get_project(args.project_id)
        readiness = RegulatoryService(store).assess_readiness(args.project_id, persist=False)
        print(
            json.dumps(
                {
                    "project": project.model_dump(mode="json"),
                    "audit_chain_valid": store.verify_audit_chain(args.project_id),
                    "regulatory_readiness": readiness.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
