def FAILURE_STAGE
pipeline {
   agent any

   stages {
      stage('Clone repo') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            cleanWs()
            git credentialsId: 'gitlab', url: 'https://gitlab.seis.exa2pro.iti.gr/exa2pro/starpu.git', branch: 'master'
            
         }
      }
      stage('Build') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            sh '''
                ls
                ./autogen.sh
                ./configure --prefix=/home/theioak/.jenkins/workspace/starpu_pipeline/lib --disable-build-examples
                make
                make install
                '''
         }
      }
      stage('Test') {
         steps {
            script{
                FAILURE_STAGE=env.STAGE_NAME
            }
            sh '''
                pwd
                ls
                git branch
                find . -type f -name Makefile -exec sed -i "s:MPIEXEC_ARGS = :MPIEXEC_ARGS = --allow-run-as-root:" {} \\;
                make check
                '''
         }
      }
   }
   post {
      success {
        mail bcc: '', body: "<b>Build status: Success</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: starpu <br> Branch: ft_checkpoint", cc: '', from: 'jenkins@gitlab.seis.iti.gr', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION: Project name -> ${env.JOB_NAME}", to: 'theioak@iti.gr,samuel.thibault@inria.fr', charset: 'UTF-8', mimeType: 'text/html'
      }
      failure {
        mail bcc: '', body: "<b>Build status: Failed</b><br>Jenkins pipeline: ${env.JOB_NAME} <br>Build Number: ${env.BUILD_NUMBER} <br> GitLab project: starpu <br> Branch: ft_checkpoint", cc: '', charset: 'UTF-8', from: 'jenkins@gitlab.seis.iti.gr', mimeType: 'text/html', replyTo: '', subject: "JENKINS EMAIL NOTIFICATION (CI ERROR): Project name -> ${env.JOB_NAME}", to: "theioak@iti.gr,samuel.thibault@inria.fr"
      }
   }

}
