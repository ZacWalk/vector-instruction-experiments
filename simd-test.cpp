// simd-test.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <windows.h>
#include <iostream>
#include <intrin.h>


#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#define COMPILE_SIMD_INTRINSIC
#endif

#if defined(_MSC_VER) && defined(_M_ARM64)
#define COMPILE_ARM_INTRINSIC
#endif

static uint64_t now_ns()
{
	LARGE_INTEGER tps = { 0 };
	QueryPerformanceFrequency(&tps);

	LARGE_INTEGER pc = { 0 };
	QueryPerformanceCounter(&pc);
	return (pc.QuadPart * 1000000000ll) / tps.QuadPart;
}

static BOOL sse2supported = ::IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE);
static BOOL neon2supported = ::IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE);

static constexpr size_t hash_size = 64;

struct vector64_t
{
	union
	{
#ifdef COMPILE_SIMD_INTRINSIC
		__m128i mm[hash_size / 16];
#endif
#ifdef COMPILE_ARM_INTRINSIC
		uint8x16_t nn[hash_size / 16];
#endif
		uint8_t h[hash_size];
	};
};

__forceinline uint32_t distance_c(const vector64_t* v1, const vector64_t* v2)
{
	uint32_t distance = 0;

	for (int i = 0; i < hash_size; i++)
	{
		const auto dd = v1->h[i] - v2->h[i];
		distance += abs(dd);
	}

	return distance;
}

__forceinline uint32_t distance_sse(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_SIMD_INTRINSIC
	if (sse2supported)
	{
		__m128i dist = _mm_sad_epu8(v1->mm[0], v2->mm[0]);
		dist = _mm_adds_epi16(dist, _mm_sad_epu8(v1->mm[1], v2->mm[1]));
		dist = _mm_adds_epi16(dist, _mm_sad_epu8(v1->mm[2], v2->mm[2]));
		dist = _mm_adds_epi16(dist, _mm_sad_epu8(v1->mm[3], v2->mm[3]));
		return dist.m128i_i16[0] + dist.m128i_i16[4];
	}	
#endif

	return 0;
}

__forceinline uint32_t distance_neon(const vector64_t* v1, const vector64_t* v2)
{
#ifdef COMPILE_ARM_INTRINSIC
	if (neon2supported)
	{
		uint8x16_t dist = vabdq_u8(v1->nn[0], v2->nn[0]);
		dist = vqaddq_u16(dist, vabdq_u8(v1->nn[1], v2->nn[1]));
		dist = vqaddq_u16(dist, vabdq_u8(v1->nn[2], v2->nn[2]));
		dist = vqaddq_u16(dist, vabdq_u8(v1->nn[3], v2->nn[3]));
		auto result = vpaddlq_u32(vpaddlq_u16(dist));
		return result.n128_i64[0] + result.n128_i64[1];
	}	
#endif

	return 0;
}

const vector64_t* make_hash(const char* src)
{
	auto result = (vector64_t*)_aligned_malloc(hash_size, 16);
	for (int i = 0; i < hash_size; i++) result->h[i] = src[i];
	return result;
}

int main()
{
	const auto v1 = make_hash("bq0zgkfbNEhAzGQ2V2W0stbpqQyQ04zrF0TgxmVoJf9O5Wk65EghJBca378cCggd");
	const auto v2 = make_hash("e0MiFoM5x53XfZrCCKuH1VovqgJatp2qTR6q9UZwHkhAszSnztPzTlhTHR2xiA41");

	// calculate distance various implimentations
	const auto dc = distance_c(v1, v2);
	const auto dsse = distance_sse(v1, v2);
	const auto dneon = distance_neon(v1, v2);

	const auto timing_iterations = 1000000ull;
	auto total_difference = 0ull; // needed to avoid opptimizing out code	
	
	// measure time of [timing_iterations] distance calulations
	auto start_time = now_ns();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_c(v1, v2);
	}
	const auto time_c = now_ns() - start_time;

	start_time = now_ns();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_sse(v1, v2);
	}
	const auto time_sse = now_ns() - start_time;

	start_time = now_ns();
	for (int i = 0ull; i < timing_iterations; i++)
	{
		total_difference += distance_neon(v1, v2);
	}
	const auto time_neon = now_ns() - start_time;

	// report results
    std::cout << "Calc distance C:    " <<  dc << " (" << time_c << " nanoseconds) " << std::endl;
	std::cout << "Calc distance SSE:  " <<  dsse << " (" << time_sse << " nanoseconds) " << std::endl;
	std::cout << "Calc distance NEON: " <<  dneon << " (" << time_neon << " nanoseconds) " << std::endl;
	std::cout << "Total difference: " << total_difference << std::endl;

	return 0;
}
