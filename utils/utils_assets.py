import os

from cryptography.fernet import Fernet


def og_decrypt_file(encrypted_filename, decrypted_filename):
    with open(os.environ["KEY_PATH"], "rb") as filekey:
        key = filekey.read()
    fernet = Fernet(key)

    with open(encrypted_filename, "rb") as enc_f:
        encrypted = enc_f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(decrypted_filename, "wb") as decrypted_file:
        decrypted_file.write(decrypted)
