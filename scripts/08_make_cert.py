#!/usr/bin/env python
"""Create a self-signed HTTPS certificate so a phone can use the camera over the LAN.

  python scripts/08_make_cert.py                # certs/cert.pem + certs/key.pem
  uvicorn server.app:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem

Then open https://<this-PC-LAN-IP>:8443 on the phone and accept the certificate warning once.
(Browsers only allow camera access on https or localhost.) certs/ is git-ignored.
"""
import datetime, ipaddress, socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

out = Path(__file__).resolve().parents[1] / "certs"; out.mkdir(exist_ok=True)
ips = {"127.0.0.1"}
try:
    ips |= {i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
except OSError:
    pass
key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "strand-inspection")])
now = datetime.datetime.now(datetime.timezone.utc)
cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(
            [x509.DNSName("localhost")] + [x509.IPAddress(ipaddress.ip_address(i)) for i in sorted(ips)]), False)
        .sign(key, hashes.SHA256()))
(out / "key.pem").write_bytes(key.private_bytes(serialization.Encoding.PEM,
    serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
(out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
print("wrote", out / "cert.pem", "and key.pem; valid for", ", ".join(sorted(ips)))
