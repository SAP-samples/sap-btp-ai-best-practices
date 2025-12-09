sap.ui.define([
    "sap/fe/test/JourneyRunner",
	"videoincidentmonitor/test/integration/pages/MediaAnalysisList",
	"videoincidentmonitor/test/integration/pages/MediaAnalysisObjectPage"
], function (JourneyRunner, MediaAnalysisList, MediaAnalysisObjectPage) {
    'use strict';

    var runner = new JourneyRunner({
        launchUrl: sap.ui.require.toUrl('videoincidentmonitor') + '/test/flpSandbox.html#videoincidentmonitor-tile',
        pages: {
			onTheMediaAnalysisList: MediaAnalysisList,
			onTheMediaAnalysisObjectPage: MediaAnalysisObjectPage
        },
        async: true
    });

    return runner;
});

