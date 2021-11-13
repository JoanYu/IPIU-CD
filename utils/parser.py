import argparse
import json
try:
    import config
except:
    from utils import config

def get_parser_with_args():
    parser = argparse.ArgumentParser(description='Training change detection network')
    metadata = config.Config
    parser.set_defaults(**metadata)
    return parser, metadata

if __name__ == '__main__':
    parser, metadata = get_parser_with_args()
    print(parser, metadata)
    print('okk!')