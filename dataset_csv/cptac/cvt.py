import os

study='UCEC'
raw_data = 'raw_{}.csv'.format(study)

save_handle = open('{}.csv'.format(study), 'w')
save_handle.write('case_id,survival_months,censorship,slide_id\n')

# parser avaiable slides
ava_slide_path = f'../temporty_csv/CPTAC__{study}.csv'
with open(ava_slide_path) as f:
    _ = f.readline()
    slide_ids = {}
    for line in f:
        _, sid, _, _ = line.split(',')
        # BRCA
        if study in ['BRCA', 'COAD']:
            case_id = sid.split('-')[0]
        elif study in ['LUAD', 'GBM', 'UCEC']:
            case_id = '-'.join(sid.split('-')[:2])
        else:
            raise NotImplementedError
        
        if case_id in slide_ids.keys():
            slide_ids[case_id].append(sid)
        else:
            slide_ids[case_id] = [sid]

with open(raw_data) as f:
    _ = f.readline()
    for line in f:
        case_id, s_days, event, _, _ = line.split(',')
        try:
            survival_months = float(s_days)/30
            slide_id = slide_ids[case_id]
            if len(slide_id) > 1:
                slide_id = ';'.join(slide_id)
            else:
                slide_id = slide_id[0]
                
            save_handle.write('{},{},{},{}\n'.format(case_id, survival_months, event, slide_id))
        except:
            print('failed to add:', line)
        
        