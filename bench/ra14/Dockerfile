# RA14: prevent-vs-detect for exfiltration, and the real cost of the diff (the review's item #6).
# One small image serves the victim, the attacker listener, and the allowlisted metrics sink.
# iptables is the real pre-commit egress enforcement (an allowlist); busybox nc is the channel.
FROM alpine:latest
RUN apk add --no-cache iptables
CMD ["sleep", "3600"]
