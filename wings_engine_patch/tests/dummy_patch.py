
PATCH_APPLIED = False
LAST_CALLED_ARGS = None

def mock_patch_func():
    global PATCH_APPLIED
    PATCH_APPLIED = True

def reset():
    global PATCH_APPLIED
    PATCH_APPLIED = False
