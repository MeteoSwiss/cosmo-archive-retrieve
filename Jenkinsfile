class Globals {
    // the library version
    static String version = 'latest'

    // the default python version
    static String pythonVersion = '3.11'

    // the tag used when publishing documentation
    static String documentationTag = ''
}

@Library('dev_tools@main') _
pipeline {
    agent {label 'podman'}

    parameters {
        booleanParam(name: 'RELEASE_BUILD', defaultValue: false, description: 'Creates and publishes a new release')
        booleanParam(name: 'PUBLISH_DOCUMENTATION', defaultValue: false, description: 'Publishes the generated documentation')
    }

    environment {
        PROJECT_NAME = 'cosmo-archive-retrieve'

        PIP_USER = 'python-mch'
        SCANNER_HOME = tool name: 'Sonarqube-certs-PROD', type: 'hudson.plugins.sonar.SonarRunnerInstallation';

        HTTP_PROXY = 'http://proxy.meteoswiss.ch:8080'
        HTTPS_PROXY = 'http://proxy.meteoswiss.ch:8080'
        NO_PROXY = '.meteoswiss.ch,localhost'
    }

    options {
        gitLabConnection('CollabGitLab')

        // New jobs should wait until older jobs are finished
        disableConcurrentBuilds()
        // Discard old builds
        buildDiscarder(logRotator(artifactDaysToKeepStr: '7', artifactNumToKeepStr: '1', daysToKeepStr: '45', numToKeepStr: '10'))
        // Timeout the pipeline build after 1 hour
        timeout(time: 1, unit: 'HOURS')
    }

    stages {
        stage('Init') {
            steps {
                updateGitlabCommitStatus name: 'Build', state: 'running'
                script {
                    Globals.documentationTag = env.BRANCH_NAME
                }
            }
        }
        stage('Regex') {
            steps {
                sh '''
                    sed -i "s/service.meteoswiss.ch/hub.meteoswiss.ch/g" pyproject.toml
                '''
            }
        }
        stage('Test') {
            parallel {
                stage('3.10') {
                    steps {
                        script {
                            runWithPodman.poetryPytest '3.10', false, false
                        }
                    }
                }
                // python 3.11 is the default version, used for executing pylint, mypy, sphinx etc.
                // all libs. are kept in the .venv folder
                stage('python 3.11') {
                    steps {
                        script {
                            runWithPodman.poetryPytest Globals.pythonVersion
                        }
                    }
                }
            }
            post {
                always {
                    junit keepLongStdio: true, testResults: 'junit*.xml'
                }
            }
        }

        stage('Run Pylint') {
            steps {
                script {
                    runWithPodman.pythonCmd Globals.pythonVersion,
                        'poetry run pylint -rn --output-format=parseable --output=pylint.log --exit-zero cosmo_archive_retrieve'
                }
            }
        }

        // myPy is treated inside Jenkins because it is not yet integrated with SonarQube (the rest of the CI results is published therein)
        stage('Run Mypy') {
            steps {
                script {
                    runWithPodman.pythonCmd Globals.pythonVersion,
                        'poetry run mypy -p cosmo_archive_retrieve | grep error | tee mypy.log'
                    recordIssues(qualityGates: [[threshold: 10, type: 'TOTAL', unstable: false]], tools: [myPy(pattern: 'mypy.log')])
                }
            }
            post {
                failure {
                    script {
                        error "Too many mypy issues, exiting now..."
                    }
                }
            }
        }

        stage('SonarQube Analysis') {
            steps {
                withSonarQubeEnv("Sonarqube-PROD") {
                    // fix source path in coverage.xml
                    // (required because coverage is calculated using podman which uses a differing file structure)
                    // https://stackoverflow.com/questions/57220171/sonarqube-client-fails-to-parse-pytest-coverage-results
                    sh "sed -i 's/\\/src\\/app-root/.\\//g' coverage.xml"
                    sh "${SCANNER_HOME}/bin/sonar-scanner"
                }
            }
        }

        stage("Quality Gate") {
            steps {
                timeout(time: 1, unit: 'HOURS') {
                    // Parameter indicates whether to set pipeline to UNSTABLE if Quality Gate fails
                    // true = set pipeline to UNSTABLE, false = don't
                    waitForQualityGate abortPipeline: true
                }
            }
        }

        stage('Release') {
            when { expression { params.RELEASE_BUILD } }
            steps {
                echo 'Build a wheel and publish'
                script {
                    withCredentials([string(credentialsId: "python-mch-nexus-secret", variable: 'PIP_PWD')]) {
                        runDevScript("build/poetry-lib-release.sh ${env.PIP_USER} $PIP_PWD 3.11")
                        Globals.version = sh(script: 'git describe --tags --abbrev=0', returnStdout: true).trim()
                        Globals.documentationTag = Globals.version
                        env.TAG_NAME = Globals.documentationTag
                    }
                }
            }
        }

        stage('Build Documentation') {
            when { expression { params.PUBLISH_DOCUMENTATION } }
            steps {
                script {
                    runWithPodman.pythonCmd Globals.pythonVersion,
                        'poetry install && poetry run sphinx-build doc doc/_build'
                }
            }
        }

        stage('Publish Documentation') {
            when { expression { params.PUBLISH_DOCUMENTATION } }
            environment {
                PATH = "$HOME/tools/openshift-client-tools:$PATH"
                KUBECONFIG = "$workspace/.kube/config"
            }
            steps {
                withCredentials([string(credentialsId: "documentation-main-prod-token", variable: 'TOKEN')]) {
                    sh "oc login https://api.cp.meteoswiss.ch:6443 --token \$TOKEN"
                    publishDoc 'doc/_build/', env.PROJECT_NAME, Globals.version, 'python', Globals.documentationTag
                }
            }
            post {
                cleanup {
                    sh 'oc logout || true'
                }
            }
        }
    }

    post {
        aborted {
            updateGitlabCommitStatus name: 'Build', state: 'canceled'
        }
        failure {
            updateGitlabCommitStatus name: 'Build', state: 'failed'
            echo 'Sending email'
            emailext(subject: "${currentBuild.fullDisplayName}: ${currentBuild.currentResult}",
                attachLog: true,
                attachmentsPattern: 'generatedFile.txt',
                body: "Job '${env.JOB_NAME} #${env.BUILD_NUMBER}': ${env.BUILD_URL}",
                recipientProviders: [requestor(), developers()])
        }
        success {
            echo 'Build succeeded'
            updateGitlabCommitStatus name: 'Build', state: 'success'
        }
        always{
            echo 'Cleaning up workspace'
            deleteDir()
        }
    }
}
