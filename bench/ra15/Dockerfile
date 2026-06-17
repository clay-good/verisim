# RA15 the covert-channel frontier: a port/host egress allowlist (RA14's prevent arm) is NOT covering
# for exfiltration, because a legitimate, allowed channel (DNS) carries arbitrary data. dnsmasq is the
# attacker's logging nameserver; bind-tools gives nslookup; iptables is the egress policy; busybox nc
# is the direct-TCP channel. One image serves both the attacker resolver and the victim.
FROM alpine:latest
RUN apk add --no-cache dnsmasq bind-tools iptables
CMD ["sleep", "3600"]
