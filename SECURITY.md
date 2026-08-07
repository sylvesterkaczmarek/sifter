# Security Policy

## Reporting a vulnerability

**Please do not open a public issue or pull request for a security problem, and
please do not post details in a public channel.**

Report it by email to **aisi.security@dsit.gov.uk**. That mailbox is monitored by
the UK AI Security Institute's security team and is the route that works for
anyone, inside or outside AISI.

If the Security tab offers a *Report a vulnerability* button, you may use that
as an equivalent private reporting route.

### What to include

- What an attacker can do, and what they need in order to do it (network access,
  a repository you can get someone to clone, an account on a shared filesystem).
- The smallest reproduction you have — a `sifter.yaml`, a command, a registry
  reference. Please don't attach real credentials.
- The sifter version (`sifter version`) and how it is installed.

We will acknowledge your report and, once we have reproduced the problem, agree a
remediation timeline with you. Please give us a reasonable chance to ship a fix
before disclosing publicly; we are happy to credit you when we do, unless you
would rather we didn't.

## Scope

In scope: anything in this repository — the `sifter` CLI, its configuration
handling, the shell fragments it generates, and its build and publish paths.

Of particular interest, because these are the trust boundaries the tool depends
on:

- **A cloned repository's `sifter.yaml` is untrusted input.** It must not be able
  to choose a binary sifter executes, relax signature verification, or otherwise
  act on the machine of whoever runs sifter inside the clone. The user config
  (`~/.config/sifter/config.yaml`) and `SIFTER_*` environment variables are
  trusted; the project file is not. In the `oci:` section it may set only the keys
  README lists as project-settable; a way to make it contribute any other key is
  a vulnerability, not a feature request.
- **Signature enforcement on the OCI/ECR path.** Anything that lets an unsigned
  or wrongly-signed image be pulled, cached or run as if it had been verified.
- **Command construction.** Sifter emits shell for Apptainer, `oras`, `cosign`
  and SLURM; injection through a manifest, image name or registry reference is
  in scope.
- **Local path containment.** A manifest-supplied build key must not make a
  tagged image, cache entry or log escape the configured sifter directories.

## Known limitations (not vulnerabilities)

These are documented behaviours rather than bugs. Reports about them are welcome
as design feedback, but they are not treated as vulnerabilities:

- **The shared-filesystem and S3 registry backends are unsigned.** They cannot
  cosign, and `SIFTER_SIGN`/`SIFTER_VERIFY` have no effect on them; images there
  are trusted on the strength of filesystem or bucket permissions alone. The
  configuration summary says so whenever one of them is the active backend. Use
  an OCI/ECR registry where signatures matter.
- **A project file fully controls the `registry:` section.** A repository may
  choose the shared-filesystem path or the S3 bucket sifter reads and writes when
  you run it inside that repository. Those backends have no signing to weaken, and
  a repository naming its own storage is the point of the section; treat one you
  didn't write with the same care as any other code you clone.
- **A project file may name the OCI registries used inside its own clone.**
  `oci.registries` stays project-settable, so a repository can point `sifter push`
  at a registry of its choosing — with your credentials, if you happen to hold
  them for it. It cannot weaken signing to do so: the image is still signed with
  your key and verified against it on pull, and a project file offering *no*
  registries is treated as one that said nothing rather than as a licence to drop
  you onto an unsigned backend. Check where a repository publishes
  before you push from a clone you didn't write.
- **Sifter trusts the container images a manifest names.** It builds what you ask
  it to build; a malicious base image is a supply-chain problem upstream of
  sifter, not a flaw in it.
- **A user who can already run arbitrary commands on the machine can subvert
  sifter.** Anything reachable from `SIFTER_*`, the user config, or the local
  filesystem is inside the trust boundary by design.
