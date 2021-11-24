mkdir -p ../data
cd ../data
wget https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2
wget https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2
wget https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2
tar -xf 20050311_spam_2.tar.bz2
tar -xf 20030228_easy_ham_2.tar.bz2
tar -xf 20030228_hard_ham.tar.bz2
