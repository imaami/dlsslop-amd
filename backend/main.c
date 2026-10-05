/** @file
 *
 * Native Linux worker for DLSS5VKLayer's shared-memory transport: dlsslopd.
 */
// SPDX-License-Identifier: MIT
#include <stdint.h>
#include <stdio.h>

#include "error.h"
#include "open.h"
#include "options.h"
#include "serve.h"
#include "shm_protocol.h"

/** @brief Does what the options ask for: --diagnose, --self-test, the offline mode, or serving.
 *
 * @param o The options.
 * @param e Receives the words for what failed, or nullptr.
 * @return  ERROR_NONE, or the code of what failed.
 */
static enum error_code
run (struct options *o,
     struct error   *e)
{
	uint32_t const tier = o->tier ? o->tier : kNativeDefaultTier;
	if (o->diagnose)
		return open_run_engine(o, tier, nullptr, e);
	if (o->test_identity)
		fprintf(stderr, "IDENTITY TEST MODE: no model, no HIP, no neural rendering.\n");
	if (o->self_test || o->input_length)
		return open_run_engine(o, tier, nullptr, e);
	return serve_run(o, e);
}

int
main (int    argc,
      char **argv)
{
	struct options o;
	struct error e;
	if (options_parse(&o, argc, argv, &e)) {
		fprintf(stderr, "dlsslopd: %s\n", e.what);
		return 1;
	}
	// --help is printed already.
	int status = 0;
	if (!o.help && run(&o, &e)) {
		fprintf(stderr, "dlsslopd: %s\n", e.what);
		status = 1;
	}
	options_fini(&o);
	return status;
}
