"""Prepare and run a baseline EM comparison only in the isolated example copy."""
from pathlib import Path
import json
import os
import traceback

WORK=Path('D:/Downloads/ADS_XAI_runs/experiments/em_probe_20261024')
WS=WORK/'source_example/mw_filter_wrk'


def create_view(workspace_path):
    import keysight.ads.de as de
    import keysight.ads.emtools as em
    workspace=de.open_workspace(workspace_path)
    cell=de.Library.get('mw_filter_lib').cell('cmstpfil')
    substrate=em.get_substrate_info(('mw_filter_lib','cmstpfil','emSetup'))
    if not cell.view_exists('rfpro'):
        em.create_empro_view(('mw_filter_lib','cmstpfil','rfpro'),'rfpro',('mw_filter_lib','cmstpfil','layout'),tuple(substrate))
    workspace.close()
    return {'rfpro_view_created':True,'substrate':substrate}


def prepare_and_run(workspace_path):
    import empro
    import empro.toolkit.analysis
    from keysight.edatoolbox import xxpro,ads
    xxpro.use_workspace(workspace_path)
    xxpro.load_pro_view(ads.LibraryCellView(library='mw_filter_lib',cell='cmstpfil',view='rfpro'))
    with empro.activeProject as project:
        analysis=empro.analysis.Analysis.fromEmSetup('emSetup')
        project.analyses.clear();project.analyses.append(analysis)
        project.saveActiveProject()
        attrs={k:str(getattr(analysis,k)) for k in ['name','analysisType'] if hasattr(analysis,k)}
        print('EM_ANALYSIS_PREPARED '+str(attrs),flush=True)
    # Exit the project-update transaction before creating simulations.
    empro.toolkit.analysis.runAnalysis(empro.activeProject.analyses[-1],waitForConfirmation=False,saveProject=True)
    print('EM_ANALYSIS_SUBMITTED',flush=True)
    from empro.toolkit import simulation
    simulation.wait(completionTimeSeconds=300)
    return {'analysis':attrs,'simulations':[{'status':str(s.status),'path':str(s.simulationPath())} for s in empro.activeProject.simulations]}


def main():
    os.environ['HPEESOF_DIR']='E:/ADS2026'
    from keysight.edatoolbox import multi_python
    previous=WORK/'rfpro_probe_result.json'
    if previous.exists():
        import shutil
        count=len(list(WORK.glob('rfpro_probe_previous_*.json')))+1
        shutil.copy2(previous,WORK/f'rfpro_probe_previous_{count:02d}.json')
    result={'status':'started','workspace':str(WS),'source_type':'fullwave attempt; not verified until solver completion and exported data checks'}
    try:
        with multi_python.ads_context() as caller:result['prepare']=caller.call(create_view,args=[str(WS)])
        print('RFPRO_VIEW_PREPARED',flush=True)
        with multi_python.xxpro_context() as caller:result['run']=caller.call(prepare_and_run,args=[str(WS)])
        result['status']='returned_for_result_verification'
    except Exception as error:
        result.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    (WORK/'rfpro_probe_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
