#ifndef NGSPICE_NI_GMRES_SHA256_H
#define NGSPICE_NI_GMRES_SHA256_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char buffer[64];
    size_t buffer_length;
} ngspice_gmres_sha256_t;

static const uint32_t ngspice_gmres_sha256_constants[64] = {
    UINT32_C(0x428a2f98), UINT32_C(0x71374491),
    UINT32_C(0xb5c0fbcf), UINT32_C(0xe9b5dba5),
    UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
    UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5),
    UINT32_C(0xd807aa98), UINT32_C(0x12835b01),
    UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
    UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe),
    UINT32_C(0x9bdc06a7), UINT32_C(0xc19bf174),
    UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
    UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc),
    UINT32_C(0x2de92c6f), UINT32_C(0x4a7484aa),
    UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
    UINT32_C(0x983e5152), UINT32_C(0xa831c66d),
    UINT32_C(0xb00327c8), UINT32_C(0xbf597fc7),
    UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
    UINT32_C(0x06ca6351), UINT32_C(0x14292967),
    UINT32_C(0x27b70a85), UINT32_C(0x2e1b2138),
    UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
    UINT32_C(0x650a7354), UINT32_C(0x766a0abb),
    UINT32_C(0x81c2c92e), UINT32_C(0x92722c85),
    UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
    UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3),
    UINT32_C(0xd192e819), UINT32_C(0xd6990624),
    UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
    UINT32_C(0x19a4c116), UINT32_C(0x1e376c08),
    UINT32_C(0x2748774c), UINT32_C(0x34b0bcb5),
    UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
    UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3),
    UINT32_C(0x748f82ee), UINT32_C(0x78a5636f),
    UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
    UINT32_C(0x90befffa), UINT32_C(0xa4506ceb),
    UINT32_C(0xbef9a3f7), UINT32_C(0xc67178f2)
};

static uint32_t
ngspice_gmres_sha256_rotr(uint32_t value, unsigned int count)
{
    return (value >> count) | (value << (32U - count));
}

static void
ngspice_gmres_sha256_transform(
    ngspice_gmres_sha256_t *context,
    const unsigned char block[64]
)
{
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0U; index < 16U; index++) {
        size_t offset = index * 4U;
        words[index] =
            ((uint32_t) block[offset] << 24U) |
            ((uint32_t) block[offset + 1U] << 16U) |
            ((uint32_t) block[offset + 2U] << 8U) |
            (uint32_t) block[offset + 3U];
    }
    for (index = 16U; index < 64U; index++) {
        uint32_t value15 = words[index - 15U];
        uint32_t value2 = words[index - 2U];
        uint32_t sigma0 =
            ngspice_gmres_sha256_rotr(value15, 7U) ^
            ngspice_gmres_sha256_rotr(value15, 18U) ^
            (value15 >> 3U);
        uint32_t sigma1 =
            ngspice_gmres_sha256_rotr(value2, 17U) ^
            ngspice_gmres_sha256_rotr(value2, 19U) ^
            (value2 >> 10U);
        words[index] =
            words[index - 16U] + sigma0 +
            words[index - 7U] + sigma1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0U; index < 64U; index++) {
        uint32_t sigma1 =
            ngspice_gmres_sha256_rotr(e, 6U) ^
            ngspice_gmres_sha256_rotr(e, 11U) ^
            ngspice_gmres_sha256_rotr(e, 25U);
        uint32_t choose = (e & f) ^ ((~e) & g);
        uint32_t temporary1 =
            h + sigma1 + choose +
            ngspice_gmres_sha256_constants[index] + words[index];
        uint32_t sigma0 =
            ngspice_gmres_sha256_rotr(a, 2U) ^
            ngspice_gmres_sha256_rotr(a, 13U) ^
            ngspice_gmres_sha256_rotr(a, 22U);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sigma0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void
ngspice_gmres_sha256_init(ngspice_gmres_sha256_t *context)
{
    context->state[0] = UINT32_C(0x6a09e667);
    context->state[1] = UINT32_C(0xbb67ae85);
    context->state[2] = UINT32_C(0x3c6ef372);
    context->state[3] = UINT32_C(0xa54ff53a);
    context->state[4] = UINT32_C(0x510e527f);
    context->state[5] = UINT32_C(0x9b05688c);
    context->state[6] = UINT32_C(0x1f83d9ab);
    context->state[7] = UINT32_C(0x5be0cd19);
    context->bit_count = UINT64_C(0);
    context->buffer_length = 0U;
}

static void
ngspice_gmres_sha256_update(
    ngspice_gmres_sha256_t *context,
    const void *data,
    size_t length
)
{
    const unsigned char *bytes = (const unsigned char *) data;

    context->bit_count += (uint64_t) length * UINT64_C(8);
    while (length > 0U) {
        size_t available = 64U - context->buffer_length;
        size_t take = length < available ? length : available;
        memcpy(context->buffer + context->buffer_length, bytes, take);
        context->buffer_length += take;
        bytes += take;
        length -= take;
        if (context->buffer_length == 64U) {
            ngspice_gmres_sha256_transform(context, context->buffer);
            context->buffer_length = 0U;
        }
    }
}

static void
ngspice_gmres_sha256_final(
    ngspice_gmres_sha256_t *context,
    unsigned char digest[32]
)
{
    uint64_t bit_count = context->bit_count;
    size_t index;

    context->buffer[context->buffer_length++] = 0x80U;
    if (context->buffer_length > 56U) {
        while (context->buffer_length < 64U)
            context->buffer[context->buffer_length++] = 0U;
        ngspice_gmres_sha256_transform(context, context->buffer);
        context->buffer_length = 0U;
    }
    while (context->buffer_length < 56U)
        context->buffer[context->buffer_length++] = 0U;
    for (index = 0U; index < 8U; index++) {
        context->buffer[63U - index] =
            (unsigned char) (bit_count >> (index * 8U));
    }
    ngspice_gmres_sha256_transform(context, context->buffer);

    for (index = 0U; index < 8U; index++) {
        digest[index * 4U] =
            (unsigned char) (context->state[index] >> 24U);
        digest[index * 4U + 1U] =
            (unsigned char) (context->state[index] >> 16U);
        digest[index * 4U + 2U] =
            (unsigned char) (context->state[index] >> 8U);
        digest[index * 4U + 3U] =
            (unsigned char) context->state[index];
    }
    memset(context, 0, sizeof(*context));
}

static void
ngspice_gmres_sha256_hex(
    const unsigned char digest[32],
    char output[65]
)
{
    static const char hexadecimal[] = "0123456789abcdef";
    size_t index;
    for (index = 0U; index < 32U; index++) {
        output[index * 2U] = hexadecimal[digest[index] >> 4U];
        output[index * 2U + 1U] = hexadecimal[digest[index] & 0x0fU];
    }
    output[64] = '\0';
}

static void
ngspice_gmres_sha256_update_i64_le(
    ngspice_gmres_sha256_t *context,
    int64_t value
)
{
    unsigned char bytes[8];
    uint64_t bits = (uint64_t) value;
    size_t index;
    for (index = 0U; index < 8U; index++)
        bytes[index] = (unsigned char) (bits >> (index * 8U));
    ngspice_gmres_sha256_update(context, bytes, sizeof(bytes));
}

static void
ngspice_gmres_sha256_update_f64_le(
    ngspice_gmres_sha256_t *context,
    double value
)
{
    unsigned char bytes[8];
    uint64_t bits = UINT64_C(0);
    size_t index;
    memcpy(&bits, &value, sizeof(bits));
    for (index = 0U; index < 8U; index++)
        bytes[index] = (unsigned char) (bits >> (index * 8U));
    ngspice_gmres_sha256_update(context, bytes, sizeof(bytes));
}

#ifdef NGSPICE_GMRES_SHA256_ENABLE_SELFTEST
static int
ngspice_gmres_sha256_selftest(void)
{
    static const int64_t indptr[] = {0, 0, 1, 2};
    static const int64_t indices[] = {1, 0};
    static const double values[] = {2.5, -3.0};
    static const char header[] =
        "schema=pals_csr_f64_v1\n"
        "rows=3\n"
        "cols=3\n"
        "nnz=2\n";
    static const char expected_abc[] =
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad";
    static const char expected_csr[] =
        "7e79e46d775a2d5641ae98ab0306d96f"
        "2888fbb75b58bc2f03e62d9d16304eca";
    ngspice_gmres_sha256_t context;
    unsigned char digest[32];
    char hexadecimal[65];
    size_t index;

    ngspice_gmres_sha256_init(&context);
    ngspice_gmres_sha256_update(&context, "abc", 3U);
    ngspice_gmres_sha256_final(&context, digest);
    ngspice_gmres_sha256_hex(digest, hexadecimal);
    if (strcmp(hexadecimal, expected_abc) != 0)
        return 0;

    ngspice_gmres_sha256_init(&context);
    ngspice_gmres_sha256_update(&context, header, sizeof(header) - 1U);
    for (index = 0U; index < sizeof(indptr) / sizeof(indptr[0]); index++)
        ngspice_gmres_sha256_update_i64_le(&context, indptr[index]);
    for (index = 0U; index < sizeof(indices) / sizeof(indices[0]); index++)
        ngspice_gmres_sha256_update_i64_le(&context, indices[index]);
    for (index = 0U; index < sizeof(values) / sizeof(values[0]); index++)
        ngspice_gmres_sha256_update_f64_le(&context, values[index]);
    ngspice_gmres_sha256_final(&context, digest);
    ngspice_gmres_sha256_hex(digest, hexadecimal);
    return strcmp(hexadecimal, expected_csr) == 0;
}
#endif

#endif
